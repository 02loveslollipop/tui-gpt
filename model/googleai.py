import os
import asyncio
from google import genai
from google.genai import types

MODEL = "gemma-4-26b-a4b-it"
ENV_VAR = "GOOGLE_API_KEY"
DISPLAY_NAME = "Google (Gemma 4 26B A4B)"


def create_client():
    api_key = os.getenv(ENV_VAR)
    if not api_key:
        raise ValueError(f"{ENV_VAR} not found in environment variables.")
    return genai.Client(api_key=api_key)


async def get_models(client=None):
    """Query the Google AI API for available generative models and return structured info."""
    if client is None:
        client = create_client()
    models = []
    # client.models.list() is synchronous, run in thread to stay async
    model_list = await asyncio.to_thread(lambda: list(client.models.list()))
    for m in model_list:
        # Filter to models that support generateContent
        actions = getattr(m, "supported_actions", []) or []
        if "generateContent" not in actions:
            continue
        models.append({
            "id": m.name,
            "name": getattr(m, "display_name", m.name) or m.name,
            "description": getattr(m, "description", "") or "",
            "context_window": getattr(m, "input_token_limit", None),
        })
    models.sort(key=lambda x: x["id"])
    return models


async def stream_response(client, context, renderer=None):
    messages = context.get_messages()

    # Extract system instruction and convert message format
    system_instruction = None
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            if system_instruction is None:
                system_instruction = msg["content"]
            else:
                system_instruction += "\n" + msg["content"]
        else:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    full_response = ""
    stream = await client.aio.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    async for chunk in stream:
        if chunk.text:
            full_response += chunk.text
            if renderer is not None:
                renderer.write_stream(chunk.text)
            else:
                print(chunk.text, end="", flush=True)

    if renderer is None:
        print()
    context.append("assistant", full_response)
