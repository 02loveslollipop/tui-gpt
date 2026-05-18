import os
from openai import AsyncOpenAI

MODEL = "gpt-5.4-nano-2026-03-17"
ENV_VAR = "OPENAI_API_KEY"
DISPLAY_NAME = "OpenAI (gpt-5.4-nano)"


def create_client():
    api_key = os.getenv(ENV_VAR)
    if not api_key:
        raise ValueError(f"{ENV_VAR} not found in environment variables.")
    return AsyncOpenAI(api_key=api_key)


async def get_models(client=None):
    """Query the OpenAI API for available chat models and return structured info."""
    if client is None:
        client = create_client()
    models = []
    model_list = await client.models.list()
    for m in model_list.data:
        model_id = m.id
        # Filter to GPT/chat models only
        if not any(prefix in model_id for prefix in ("gpt", "o1", "o3", "o4")):
            continue
        models.append({
            "id": model_id,
            "name": model_id,
            "description": "",
            "context_window": None,
        })
    models.sort(key=lambda x: x["id"])
    return models


async def stream_response(client, context, renderer=None):
    stream = await client.responses.create(
        model=MODEL,
        input=context.get_messages(),
        stream=True,
    )
    full_response = ""
    async for event in stream:
        if getattr(event, "type", None) == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                full_response += delta
                if renderer is not None:
                    renderer.write_stream(delta)
                else:
                    print(delta, end="", flush=True)

    if renderer is None:
        print()
    context.append("assistant", full_response)
