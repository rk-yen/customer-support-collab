import json
from typing import Literal

from pydantic import create_model


def _messages(prompt: str, customer_message: str) -> list[dict]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": customer_message},
    ]


def _build_router_decision_model(categories: list[str]):
    category_literal = Literal.__getitem__(tuple(categories))
    return create_model("RouterDecision", category=(category_literal, ...))


def run_raw_llm(client, customer_message: str, prompt: str, max_tokens: int = 220) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=_messages(prompt, customer_message),
        temperature=0.3,
        max_output_tokens=max_tokens,
    )
    return response.output_text.strip()


def run_context_agent(
    client,
    customer_message: str,
    prompt: str,
    max_tokens: int = 220,
) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=_messages(prompt, customer_message),
        temperature=0.3,
        max_output_tokens=max_tokens,
    )
    return response.output_text.strip()


def run_router_structured(client, customer_message: str, prompt: str, categories: list[str]) -> dict:
    router_decision_model = _build_router_decision_model(categories)
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=_messages(prompt, customer_message),
        temperature=0,
        max_output_tokens=80,
        text_format=router_decision_model,
    )
    parsed = response.output_parsed
    if parsed is None:
        content = response.output_text.strip()
        payload = {"category": content}
        category = content
    else:
        payload = parsed.model_dump()
        category = payload.get("category", "")

    if category not in categories:
        payload["category"] = "escalation" if "escalation" in categories else categories[0]
        payload["fallback_reason"] = f"Invalid category returned: {category}"
    return payload
