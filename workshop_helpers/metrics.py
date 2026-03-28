import json
import re

from arize.experiments.evaluators.base import EvaluationResult, Evaluator

IDEAL_BOUNDARY_RESPONSE = (
    "Hi Elena, I am sorry to be the bearer of difficult news - your order falls just "
    "outside our 30-day return window by 5 days, so I am unable to approve the return "
    "directly. I completely understand the frustration, especially as the item is still "
    "sealed. I can escalate this to a supervisor who has the ability to make exceptions "
    "in cases like this - would you like me to do that?"
)

TONE_JUDGE = (
    "You evaluate customer support responses for tone.\n\n"
    "GOOD: Warm, empathetic, professional.\n"
    "ACCEPTABLE: Polite but generic or slightly flat.\n"
    "POOR: Cold, robotic, or dismissive.\n\n"
    "Respond with exactly one word: Good, Acceptable, or Poor."
)

RESOLUTION_JUDGE = (
    "Compare a support response to an ideal response.\n\n"
    "Ideal:\n{ideal}\n\nActual:\n{actual}\n\n"
    "GOOD: Core issue addressed accurately and completely. "
    "If the case clearly supports an immediate resolution, the response resolves it instead of deferring.\n"
    "ACCEPTABLE: Mostly correct, but missing key specifics or stopping at next steps when it could have resolved.\n"
    "POOR: Not addressed, generic, asks for information that should already be available, or promises action without resolving.\n\n"
    "Respond with exactly one word: Good, Acceptable, or Poor."
)

LABEL_SCORE = {"Good": 1.0, "Acceptable": 0.5, "Poor": 0.0, "PASS": 1.0, "FAIL": 0.0}


def check_source_grounding(output: str, source_data: dict) -> tuple[bool, str]:
    source_str = json.dumps(source_data)
    ai_amounts = set(re.findall(r"\$[\d,]+\.?\d*", output))
    src_numbers = set(re.findall(r"[\d]+\.?\d*", source_str))
    hallucinated = [amount for amount in ai_amounts if re.sub(r"[\$,]", "", amount) not in src_numbers]
    return (not hallucinated, "OK" if not hallucinated else f"Invented: {hallucinated}")


def judge_tone(client, output: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": TONE_JUDGE},
            {"role": "user", "content": f"Response:\n{output}"},
        ],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().capitalize()


def judge_resolution(client, output: str, ideal: str = IDEAL_BOUNDARY_RESPONSE) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": RESOLUTION_JUDGE.format(ideal=ideal, actual=output)}],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().capitalize()


def composite_score(grounding_ok: bool, tone: str, resolved: str) -> float:
    return round((1.0 if grounding_ok else 0.0) + LABEL_SCORE.get(tone, 0.0) + LABEL_SCORE.get(resolved, 0.0), 1)


def score_single_response(client, output: str, source_data: dict, ideal: str = IDEAL_BOUNDARY_RESPONSE) -> dict:
    grounding_ok, grounding_detail = check_source_grounding(output, source_data)
    tone = judge_tone(client, output)
    resolved = judge_resolution(client, output, ideal=ideal)
    return {
        "grounding_ok": grounding_ok,
        "grounding_detail": grounding_detail,
        "tone": tone,
        "resolved": resolved,
        "total": composite_score(grounding_ok, tone, resolved),
    }


def compare_scores(client, outputs: dict, source_data: dict, ideal: str = IDEAL_BOUNDARY_RESPONSE) -> list[dict]:
    rows = []
    for label, output in outputs.items():
        score = score_single_response(client, output, source_data, ideal=ideal)
        rows.append({"variant": label, **score})
    return rows


class SourceGroundingEvaluator(Evaluator):
    def __init__(self, dataset_by_id: dict):
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        ok, detail = check_source_grounding(output or "", case.get("source_data", {}))
        return EvaluationResult(
            score=1.0 if ok else 0.0,
            label="PASS" if ok else "FAIL",
            explanation=detail,
        )


class ToneQualityEvaluator(Evaluator):
    def __init__(self, client):
        self.client = client

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        label = judge_tone(self.client, output or "")
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation="AI judge")


class IssueResolvedEvaluator(Evaluator):
    def __init__(self, client, dataset_by_id: dict):
        self.client = client
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        ideal = case.get("expected_output", "")
        label = judge_resolution(self.client, output or "", ideal=ideal)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation="AI judge")


def build_evaluators(client, dataset_by_id: dict) -> list[Evaluator]:
    return [
        SourceGroundingEvaluator(dataset_by_id),
        ToneQualityEvaluator(client),
        IssueResolvedEvaluator(client, dataset_by_id),
    ]
