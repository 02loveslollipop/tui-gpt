"""Unit tests for model/openai.py — targeting 100% coverage."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from classes.context import Context
from tests.conftest import AsyncIterator


# ── create_client ─────────────────────────────────────────────────────────

class TestCreateClient:
    def test_with_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            from model import openai as provider
            client = provider.create_client()
            assert client is not None

    def test_without_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            from model import openai as provider
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                provider.create_client()


# ── get_models ────────────────────────────────────────────────────────────

class TestGetModels:
    @pytest.mark.asyncio
    async def test_with_client(self, mock_openai_client):
        from model import openai as provider
        models = await provider.get_models(client=mock_openai_client)
        ids = [m["id"] for m in models]
        assert "gpt-5.4-nano-2026-03-17" in ids
        assert "o4-mini" in ids
        assert "text-embedding-ada-002" not in ids
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "description" in m
            assert "context_window" in m

    @pytest.mark.asyncio
    async def test_without_client_creates_one(self):
        from model import openai as provider
        mock_client = AsyncMock()
        model1 = MagicMock()
        model1.id = "gpt-test"
        model_list = MagicMock()
        model_list.data = [model1]
        mock_client.models.list = AsyncMock(return_value=model_list)

        with patch.object(provider, "create_client", return_value=mock_client):
            models = await provider.get_models(client=None)
            assert len(models) == 1

    @pytest.mark.asyncio
    async def test_filters_non_gpt_models(self, mock_openai_client):
        from model import openai as provider
        models = await provider.get_models(client=mock_openai_client)
        for m in models:
            assert any(p in m["id"] for p in ("gpt", "o1", "o3", "o4"))

    @pytest.mark.asyncio
    async def test_sorted_output(self, mock_openai_client):
        from model import openai as provider
        models = await provider.get_models(client=mock_openai_client)
        ids = [m["id"] for m in models]
        assert ids == sorted(ids)


# ── stream_response ───────────────────────────────────────────────────────

class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_normal_streaming(self, capsys):
        from model import openai as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        event1 = MagicMock()
        event1.type = "response.output_text.delta"
        event1.delta = "Hello "

        event2 = MagicMock()
        event2.type = "response.output_text.delta"
        event2.delta = "world!"

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(
            return_value=AsyncIterator([event1, event2])
        )

        await provider.stream_response(mock_client, ctx)

        captured = capsys.readouterr()
        assert "Hello " in captured.out
        assert "world!" in captured.out
        assert ctx.get_messages()[-1]["role"] == "assistant"
        assert ctx.get_messages()[-1]["content"] == "Hello world!"

    @pytest.mark.asyncio
    async def test_event_without_type(self, capsys):
        from model import openai as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        event1 = MagicMock(spec=[])  # no type attribute
        event2 = MagicMock()
        event2.type = "response.output_text.delta"
        event2.delta = "ok"

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(
            return_value=AsyncIterator([event1, event2])
        )

        await provider.stream_response(mock_client, ctx)
        assert ctx.get_messages()[-1]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_event_with_empty_delta(self, capsys):
        from model import openai as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        event1 = MagicMock()
        event1.type = "response.output_text.delta"
        event1.delta = ""

        event2 = MagicMock()
        event2.type = "response.output_text.delta"
        event2.delta = "content"

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(
            return_value=AsyncIterator([event1, event2])
        )

        await provider.stream_response(mock_client, ctx)
        assert ctx.get_messages()[-1]["content"] == "content"

    @pytest.mark.asyncio
    async def test_streaming_uses_renderer_when_provided(self):
        from model import openai as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        event1 = MagicMock()
        event1.type = "response.output_text.delta"
        event1.delta = "rendered"

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(
            return_value=AsyncIterator([event1])
        )
        renderer = MagicMock()

        await provider.stream_response(mock_client, ctx, renderer=renderer)

        renderer.write_stream.assert_called_once_with("rendered")
        assert ctx.get_messages()[-1]["content"] == "rendered"
