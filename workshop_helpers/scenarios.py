import json


def run_raw_llm(client, customer_message: str, prompt: str, max_tokens: int = 220) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def run_context_agent(
    client,
    customer_message: str,
    prompt: str,
    max_tokens: int = 220,
) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def run_router_structured(client, customer_message: str, prompt: str, categories: list[str]) -> dict:
    schema_hint = json.dumps({"category": categories[0]})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": customer_message,
            },
        ],
        temperature=0,
        max_tokens=80,
    )
    content = response.choices[0].message.content.strip()
    try:
        payload = json.loads(content)
        category = payload.get("category", "")
    except Exception:
        payload = {"category": content}
        category = content

    if category not in categories:
        payload["category"] = "escalation" if "escalation" in categories else categories[0]
        payload["fallback_reason"] = f"Invalid category returned: {category}"
    return payload
