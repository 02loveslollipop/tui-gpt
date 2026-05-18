import os
import asyncio
from mistralai.client import Mistral

MODEL = "mistral-small-latest"
ENV_VAR = "MISTRAL_API_KEY"
DISPLAY_NAME = "Mistral (mistral-small)"


def create_client():
    api_key = os.getenv(ENV_VAR)
    if not api_key:
        raise ValueError(f"{ENV_VAR} not found in environment variables.")
    return Mistral(api_key=api_key)


async def get_models(client=None):
    """Query the Mistral API for available models and return structured info."""
    if client is None:
        client = create_client()
    models = []
    # Use async list if available, otherwise wrap sync
    try:
        result = await asyncio.to_thread(client.models.list)
        model_list = result.data if hasattr(result, "data") else result
    except Exception:
        model_list = []
    for m in model_list:
        model_id = getattr(m, "id", str(m))
        models.append({
            "id": model_id,
            "name": getattr(m, "name", model_id) or model_id,
            "description": getattr(m, "description", "") or "",
            "context_window": getattr(m, "max_context_length", None),
        })
    models.sort(key=lambda x: x["id"])
    return models


async def stream_response(client, context, renderer=None):
    messages = context.get_messages()

    full_response = ""
    stream = await client.chat.stream_async(
        model=MODEL,
        messages=messages,
    )
    async for chunk in stream:
        delta = chunk.data.choices[0].delta.content
        if delta:
            full_response += delta
            if renderer is not None:
                renderer.write_stream(delta)
            else:
                print(delta, end="", flush=True)

    if renderer is None:
        print()
    context.append("assistant", full_response)
