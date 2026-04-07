import json
import re

from arize.experiments.evaluators.base import EvaluationResult, Evaluator

LABEL_SCORE = {"Good": 1.0, "Acceptable": 0.5, "Poor": 0.0}
VARIANT_DISPLAY = {
    "router": "router",
    "permissions": "permissions",
    "review_workflow": "review_workflow",
    "billing": "billing",
    "v2_routed": "v2_routed",
}


def _parse_judge_response(text: str) -> tuple[str, str]:
    label_match = re.search(r"LABEL:\s*(Good|Acceptable|Poor)", text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    label = label_match.group(1).capitalize() if label_match else "Poor"
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    return label, reasoning


def _parse_task_output(output: str) -> dict:
    try:
        payload = json.loads(output)
        if isinstance(payload, dict) and "response_text" in payload:
            return {
                "response_text": payload.get("response_text", ""),
                "tool_calls": payload.get("tool_calls", []),
                "action_calls": payload.get("action_calls", []),
                "metadata": payload.get("metadata", {}),
            }
    except Exception:
        pass
    return {"response_text": output or "", "tool_calls": [], "action_calls": [], "metadata": {}}


def pack_response_payload(
    response_text: str,
    tool_calls: list | None = None,
    action_calls: list | None = None,
    metadata: dict | None = None,
) -> str:
    return json.dumps(
        {
            "response_text": response_text,
            "tool_calls": tool_calls or [],
            "action_calls": action_calls or [],
            "metadata": metadata or {},
        }
    )


def normalize_text_label(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def exact_match_result(actual: str | None, expected: str | None) -> tuple[str, str]:
    normalized_actual = normalize_text_label(actual)
    normalized_expected = normalize_text_label(expected)
    if normalized_actual == normalized_expected:
        return "Good", f"Exact match on normalized label `{expected}`."
    return "Poor", f"Predicted `{actual or ''}` but expected `{expected or ''}`."


def _variant_expectation(case: dict, variant_name: str) -> str:
    category = case.get("category", "unknown")
    action_expected = case.get("action_expected", False)
    action_type = case.get("action_type", "none")
    workflow_expectation = case.get("workflow_expectation", "unknown")

    if variant_name == "permissions":
        return (
            "Prompt-only permissions specialist. Give a concise access path, explain who can grant access, "
            "and do not invent backend changes or approvals."
        )

    if variant_name == "review_workflow":
        return (
            "Context-aware workflow specialist. Use the provided workflow state and blockers specifically. "
            "Do not invent missing review steps."
        )

    if variant_name == "billing":
        if action_expected:
            return f"Tool-using billing specialist. Use tools to verify the account and complete `{action_type}` before replying."
        return "Tool-using billing specialist. Use tools to verify account and invoice details before explaining the bill."

    if variant_name == "v2_routed":
        if category == "permissions":
            return "Two-stage copilot. Route to permissions, then answer as a prompt-only permissions specialist."
        if category == "review_workflow":
            return "Two-stage copilot. Route to review_workflow, then use provided workflow context to answer."
        if category == "billing":
            if action_expected:
                return f"Two-stage copilot. Route to billing, use tools, and complete `{action_type}` before replying."
            return "Two-stage copilot. Route to billing and verify account or invoice facts with tools before replying."
        if category == "escalation":
            return "Two-stage copilot. Route to escalation, hand the case to a human, and clearly communicate the handoff."

    if workflow_expectation == "escalate":
        return "Escalate the case to a human promptly."
    if action_expected:
        return f"Complete `{action_type}` before replying."
    return "Answer the user directly and stay grounded in the available context."


def _judge_with_reasoning(client, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=140,
    )
    content = response.choices[0].message.content.strip()
    return _parse_judge_response(content)


def judge_brand_voice(client, output: str, judge_prompts: dict) -> tuple[str, str]:
    payload = _parse_task_output(output)
    return _judge_with_reasoning(client, judge_prompts["brand_voice"], f"Response:\n{payload['response_text']}")


def judge_helpfulness(client, output: str, case: dict, judge_prompts: dict) -> tuple[str, str]:
    payload = _parse_task_output(output)
    return _judge_with_reasoning(
        client,
        judge_prompts["helpfulness_system"],
        judge_prompts["helpfulness"].format(
            user_input=case.get("user_input", ""),
            ideal=case.get("expected_output", ""),
            actual=payload["response_text"],
        ),
    )


def judge_workflow_fit(
    client,
    output: str,
    case: dict,
    variant_name: str,
    variant_behavior: str,
    judge_prompts: dict,
) -> tuple[str, str]:
    payload = _parse_task_output(output)
    tool_calls = payload["tool_calls"]
    action_calls = payload["action_calls"]
    route_category = payload["metadata"].get("route_category", "")
    tool_summary = "; ".join(
        (
            f"{call.get('name', 'unknown')}({call.get('arguments', '')})"
            if isinstance(call, dict)
            else str(call)
        )
        for call in tool_calls
    ) or "none"
    action_summary = "; ".join(
        (
            f"{call.get('name', 'unknown')}({call.get('arguments', '')})"
            if isinstance(call, dict)
            else str(call)
        )
        for call in action_calls
    ) or "none"
    return _judge_with_reasoning(
        client,
        judge_prompts["workflow_system"],
        judge_prompts["workflow"].format(
            variant_name=variant_name,
            variant_behavior=variant_behavior,
            case_category=case.get("category", "unknown"),
            routed_category=route_category or "not recorded",
            variant_expectation=_variant_expectation(case, variant_name),
            workflow_expectation=case.get("workflow_expectation", "unknown"),
            missing_info_required=case.get("missing_info_required", False),
            action_expected=case.get("action_expected", False),
            action_type=case.get("action_type", "none"),
            user_input=case.get("user_input", ""),
            actual=payload["response_text"],
        )
        + f"\n\nRecorded tool calls: {tool_summary}\nRecorded action calls: {action_summary}",
    )


def _action_match_score(case: dict, output: str) -> tuple[str, str] | None:
    if not case.get("action_expected"):
        return None
    payload = _parse_task_output(output)
    expected_action = case.get("action_type")
    action_calls = payload["action_calls"]
    for call in action_calls:
        call_name = call.get("name") if isinstance(call, dict) else str(call)
        if expected_action and call_name == expected_action:
            arguments = call.get("arguments", "") if isinstance(call, dict) else ""
            return "Good", f"Recorded action `{expected_action}` was called with arguments: {arguments}"
    if action_calls:
        called_names = ", ".join(
            call.get("name", str(call)) if isinstance(call, dict) else str(call) for call in action_calls
        )
        return "Acceptable", (
            f"Recorded action calls were `{called_names}`, but none matched expected action `{expected_action}`."
        )
    return "Poor", f"No recorded action call matched expected action `{expected_action}`."


def _did_escalate(case: dict, output: str) -> tuple[bool, str]:
    payload = _parse_task_output(output)
    action_calls = payload["action_calls"]
    routed_category = payload["metadata"].get("route_category", "")
    expected_action = case.get("action_type")

    for call in action_calls:
        call_name = call.get("name") if isinstance(call, dict) else str(call)
        if call_name == "escalate_to_human":
            arguments = call.get("arguments", "") if isinstance(call, dict) else ""
            return True, f"Recorded `escalate_to_human` action call with arguments: {arguments}"

    if routed_category == "escalation" and expected_action == "escalate_to_human":
        return True, "Route category was `escalation` for a case that requires human handoff."

    return False, "No recorded `escalate_to_human` action call was found."


def escalation_decision_result(case: dict, output: str) -> tuple[str, str]:
    expected = case.get("action_type") == "escalate_to_human"
    predicted, evidence = _did_escalate(case, output)
    if expected == predicted:
        return "Good", f"Escalation decision matched expectation. {evidence}"
    return "Poor", f"Expected escalate={expected}, predicted escalate={predicted}. {evidence}"


def composite_score(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores), 1)


def score_routing_response(output: str, expected_category: str) -> dict:
    payload = _parse_task_output(output)
    label, reasoning = exact_match_result(payload["response_text"], expected_category)
    return {
        "exact_match": label,
        "exact_match_reasoning": reasoning,
        "predicted_category": payload["response_text"],
        "expected_category": expected_category,
        "total": LABEL_SCORE.get(label, 0.0),
    }


def score_single_response(
    client,
    output: str,
    case: dict,
    variant_name: str,
    variant_behavior: str,
    judge_prompts: dict,
) -> dict:
    brand_voice_label, brand_voice_reasoning = judge_brand_voice(client, output, judge_prompts=judge_prompts)
    helpfulness_label, helpfulness_reasoning = judge_helpfulness(client, output, case, judge_prompts=judge_prompts)
    workflow_label, workflow_reasoning = judge_workflow_fit(
        client,
        output,
        case,
        variant_name=variant_name,
        variant_behavior=variant_behavior,
        judge_prompts=judge_prompts,
    )

    row = {
        "brand_voice": brand_voice_label,
        "brand_voice_reasoning": brand_voice_reasoning,
        "helpfulness": helpfulness_label,
        "helpfulness_reasoning": helpfulness_reasoning,
        "workflow_fit": workflow_label,
        "workflow_reasoning": workflow_reasoning,
    }

    scores = [
        LABEL_SCORE.get(brand_voice_label, 0.0),
        LABEL_SCORE.get(helpfulness_label, 0.0),
        LABEL_SCORE.get(workflow_label, 0.0),
    ]

    action_check = _action_match_score(case, output) if variant_name in {"billing", "v2_routed"} else None
    if action_check:
        row["action_execution"] = action_check[0]
        row["action_reasoning"] = action_check[1]
        scores.append(LABEL_SCORE.get(action_check[0], 0.0))
    else:
        row["action_execution"] = ""
        row["action_reasoning"] = ""

    row["total"] = composite_score(scores)
    return row


def compare_scores(
    client,
    outputs: dict,
    case: dict,
    judge_prompts: dict,
    variant_behaviors: dict | None = None,
) -> list[dict]:
    variant_behaviors = variant_behaviors or {}
    rows = []
    for label, output in outputs.items():
        rows.append(
            {
                "variant": label,
                **score_single_response(
                    client,
                    output,
                    case,
                    variant_name=label,
                    variant_behavior=variant_behaviors.get(label, label),
                    judge_prompts=judge_prompts,
                ),
            }
        )
    return rows


class ExactMatchEvaluator(Evaluator):
    def __init__(self, expected_field: str, output_field: str = "response_text"):
        self.expected_field = expected_field
        self.output_field = output_field

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        payload = _parse_task_output(output or "")
        actual = payload.get(self.output_field, "")
        expected = dataset_row.get(self.expected_field, "")
        label, reasoning = exact_match_result(actual, expected)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class BrandVoiceEvaluator(Evaluator):
    def __init__(self, client, judge_prompts: dict):
        self.client = client
        self.judge_prompts = judge_prompts

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        label, reasoning = judge_brand_voice(self.client, output or "", judge_prompts=self.judge_prompts)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class HelpfulnessEvaluator(Evaluator):
    def __init__(self, client, dataset_by_id: dict, judge_prompts: dict):
        self.client = client
        self.dataset_by_id = dataset_by_id
        self.judge_prompts = judge_prompts

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label, reasoning = judge_helpfulness(self.client, output or "", case, judge_prompts=self.judge_prompts)
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class WorkflowFitEvaluator(Evaluator):
    def __init__(
        self,
        client,
        dataset_by_id: dict,
        variant_name: str,
        variant_behavior: str,
        judge_prompts: dict,
    ):
        self.client = client
        self.dataset_by_id = dataset_by_id
        self.variant_name = variant_name
        self.variant_behavior = variant_behavior
        self.judge_prompts = judge_prompts

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label, reasoning = judge_workflow_fit(
            self.client,
            output or "",
            case,
            variant_name=self.variant_name,
            variant_behavior=self.variant_behavior,
            judge_prompts=self.judge_prompts,
        )
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class ActionExecutionEvaluator(Evaluator):
    def __init__(self, dataset_by_id: dict):
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        result = _action_match_score(case, output or "")
        if result is None:
            return EvaluationResult(score=1.0, label="Good", explanation="No action expected for this case.")
        label, reasoning = result
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


class EscalationDecisionEvaluator(Evaluator):
    def __init__(self, dataset_by_id: dict):
        self.dataset_by_id = dataset_by_id

    def evaluate(self, dataset_row, input, output, **kwargs) -> EvaluationResult:
        case = self.dataset_by_id.get(dataset_row.get("scenario_id"), {})
        label, reasoning = escalation_decision_result(case, output or "")
        return EvaluationResult(score=LABEL_SCORE.get(label, 0.0), label=label, explanation=reasoning)


def build_evaluators(
    client,
    dataset_by_id: dict,
    variant_name: str,
    variant_behavior: str,
    judge_prompts: dict,
) -> list[Evaluator]:
    if variant_name == "router":
        return [ExactMatchEvaluator(expected_field="category")]

    evaluators: list[Evaluator] = [
        BrandVoiceEvaluator(client, judge_prompts=judge_prompts),
        HelpfulnessEvaluator(client, dataset_by_id, judge_prompts=judge_prompts),
    ]
    if variant_name == "billing":
        evaluators.append(ActionExecutionEvaluator(dataset_by_id))
    if variant_name == "v2_routed":
        evaluators.append(EscalationDecisionEvaluator(dataset_by_id))
    return evaluators
