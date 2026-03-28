def run_raw_llm(client, customer_message: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def run_context_agent(
    client,
    customer_message: str,
    prompt: str,
) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": customer_message},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

