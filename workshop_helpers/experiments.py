from datetime import datetime
import json

import pandas as pd
from arize import ArizeClient

from workshop_helpers.backend import TOOLS, build_support_agent, hydrate_backend_from_dataset, run_support_agent_threadsafe
from workshop_helpers.metrics import build_evaluators

DATASET_NAME = "cs-support-20-cases"


def dataset_index(dataset: list[dict]) -> dict:
    return {case["scenario_id"]: case for case in dataset}


def summarize_dataset(dataset: list[dict]) -> dict:
    frame = pd.DataFrame(dataset)
    return {
        "scenario_count": len(frame),
        "standard_count": int(frame[~frame.is_edge_case].shape[0]),
        "edge_case_count": int(frame[frame.is_edge_case].shape[0]),
        "categories": sorted(frame.category.unique()),
    }


def build_v2_support_snapshot(case: dict) -> dict:
    source_data = case["source_data"]
    category = case.get("category")

    snapshot = {
        "customer_name": source_data.get("customer_name"),
        "customer_id": source_data.get("customer_id"),
        "order_id": source_data.get("order_id"),
        "product_name": source_data.get("product_name"),
        "order_status": source_data.get("order_status"),
        "account_status": source_data.get("account_status"),
    }

    if category == "product":
        snapshot["product_name"] = source_data.get("product_name")

    if category == "subscription":
        snapshot["account_status"] = source_data.get("account_status")

    if category == "account_access":
        snapshot["account_status"] = source_data.get("account_status")

    return {key: value for key, value in snapshot.items() if value is not None}


def build_arize_dataframe(dataset: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": case["scenario_id"],
                "category": case["category"],
                "is_edge_case": case["is_edge_case"],
                "customer_id": case["source_data"].get("customer_id", ""),
                "user_input": case["user_input"],
                "expected_output": case["expected_output"],
            }
            for case in dataset
        ]
    )


def ensure_arize_dataset(arize_api_key: str, arize_space_id: str, dataset: list[dict]) -> dict:
    client = ArizeClient(api_key=arize_api_key)
    dataset_frame = build_arize_dataframe(dataset)
    list_response = client.datasets.list(space=arize_space_id)

    existing = next((item for item in list_response.datasets if item.name == DATASET_NAME), None)
    if existing:
        dataset_id = existing.id
        created = False
    else:
        response = client.datasets.create(
            space=arize_space_id,
            name=DATASET_NAME,
            examples=dataset_frame,
        )
        dataset_id = response.id
        created = True

    return {
        "client": client,
        "dataset_id": dataset_id,
        "dataset_name": DATASET_NAME,
        "row_count": len(dataset_frame),
        "created": created,
        "dataframe": dataset_frame,
    }


def build_tasks(client, dataset: list[dict], prompt_v1: str, prompt_v2: str) -> dict:
    cases_by_id = dataset_index(dataset)

    def task_v1(row: dict) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_v1},
                {"role": "user", "content": row["user_input"]},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def task_v2(row: dict) -> str:
        case = cases_by_id.get(row["scenario_id"])
        if not case:
            return "Error: case not found"
        support_snapshot = build_v2_support_snapshot(case)
        message = (
            f"Customer context (internal support snapshot):\n{json.dumps(support_snapshot, indent=2)}"
            f"\n\nCustomer message:\n{row['user_input']}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_v2},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def task_v3(row: dict) -> str:
        return run_support_agent_threadsafe(
            customer_message=row["user_input"],
            authenticated_customer_id=row.get("customer_id") or "UNKNOWN",
        )

    return {"task_v1": task_v1, "task_v2": task_v2, "task_v3": task_v3}


def run_experiment(arize_client, dataset_id: str, name_prefix: str, task, evaluators, concurrency: int = 3):
    experiment_name = f"{name_prefix}-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    experiment, results_df = arize_client.experiments.run(
        name=experiment_name,
        dataset=dataset_id,
        task=task,
        evaluators=evaluators,
        concurrency=concurrency,
    )
    return {"experiment": experiment, "results_df": results_df, "experiment_name": experiment_name}


def production_readiness_checklist() -> list[tuple[str, str, str]]:
    return [
        ("source_grounding >= 90% on all 20 cases", "check v3 experiment in Arize", "tools verify facts instead of relying on injected JSON"),
        ("tone Good or Acceptable >= 75%", "check v3 experiment in Arize", "adjust system prompt tone guidance if needed"),
        ("issue_resolved Good >= 60% on standard cases", "check v3 experiment in Arize", "tool coverage now includes billing, account, subscriptions"),
        ("human review gate before response sent to customer", "not yet designed", "required for v1: agent drafts, human approves"),
        ("guardrail for escalation-required cases", "take_action(escalate) creates ESC_00291", "verify the downstream ticket workflow is real, not a stub"),
        ("all case types covered by tools", "yes - profile, return, product, action", "test account access and subscription results in Arize"),
        ("sampling plan: outputs reviewed per week, by whom", "not defined", "recommended: 50 sampled per week in month 1"),
        ("threshold to advance from v1 to v2 autonomy", "not defined", "suggested: >= 90% grounding and 80% resolved Good for 4 weeks"),
    ]


def format_checklist_rows(checklist: list[tuple[str, str, str]]) -> list[str]:
    rows = [f"{'#':<3} {'CRITERION':<44} {'STATUS':<34} NOTE", "-" * 118]
    for index, (criterion, status, note) in enumerate(checklist, start=1):
        icon = "OK" if any(word in status.lower() for word in ["yes", "creates", "ok"]) else "--"
        rows.append(f"{icon} {index:<2} {criterion:<44} {status:<34} {note}")
    return rows


def prepare_experiment_bundle(client, arize_api_key: str, arize_space_id: str, dataset: list[dict], prompt_v1: str, prompt_v2: str) -> dict:
    hydrate_backend_from_dataset(dataset)
    dataset_lookup = dataset_index(dataset)
    arize_bundle = ensure_arize_dataset(arize_api_key, arize_space_id, dataset)
    return {
        **arize_bundle,
        "dataset_lookup": dataset_lookup,
        "tasks": build_tasks(client, dataset, prompt_v1, prompt_v2),
        "evaluators": build_evaluators(client, dataset_lookup),
        "tool_count": len(TOOLS),
    }
