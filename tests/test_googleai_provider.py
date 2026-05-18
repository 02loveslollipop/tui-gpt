"""Unit tests for model/googleai.py — targeting 100% coverage."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from classes.context import Context
from tests.conftest import AsyncIterator


# ── create_client ─────────────────────────────────────────────────────────

class TestCreateClient:
    def test_with_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("model.googleai.genai") as mock_genai:
                from model import googleai as provider
                client = provider.create_client()
                mock_genai.Client.assert_called_once_with(api_key="test-key")

    def test_without_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            from model import googleai as provider
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                provider.create_client()


# ── get_models ────────────────────────────────────────────────────────────

class TestGetModels:
    @pytest.mark.asyncio
    async def test_with_client_filters_generateContent(self, mock_googleai_client):
        from model import googleai as provider
        models = await provider.get_models(client=mock_googleai_client)
        ids = [m["id"] for m in models]
        assert "gemma-4-31b-it" in ids
        assert "embedding-001" not in ids

    @pytest.mark.asyncio
    async def test_model_attributes(self, mock_googleai_client):
        from model import googleai as provider
        models = await provider.get_models(client=mock_googleai_client)
        m = models[0]
        assert m["name"] == "Gemma 4 31B IT"
        assert m["description"] == "A generative model"
        assert m["context_window"] == 131072

    @pytest.mark.asyncio
    async def test_without_client_creates_one(self):
        from model import googleai as provider
        mock_client = MagicMock()
        model1 = MagicMock()
        model1.name = "test-model"
        model1.display_name = "Test"
        model1.description = "test"
        model1.supported_actions = ["generateContent"]
        model1.input_token_limit = 1000
        mock_client.models.list.return_value = iter([model1])

        with patch.object(provider, "create_client", return_value=mock_client):
            models = await provider.get_models(client=None)
            assert len(models) == 1

    @pytest.mark.asyncio
    async def test_model_missing_attributes(self):
        from model import googleai as provider
        mock_client = MagicMock()
        model1 = MagicMock(spec=["name", "supported_actions"])
        model1.name = "bare-model"
        model1.supported_actions = ["generateContent"]
        mock_client.models.list.return_value = iter([model1])

        models = await provider.get_models(client=mock_client)
        assert len(models) == 1
        assert models[0]["id"] == "bare-model"
        assert models[0]["name"] == "bare-model"

    @pytest.mark.asyncio
    async def test_model_none_supported_actions(self):
        from model import googleai as provider
        mock_client = MagicMock()
        model1 = MagicMock()
        model1.name = "null-actions"
        model1.supported_actions = None
        mock_client.models.list.return_value = iter([model1])

        models = await provider.get_models(client=mock_client)
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_sorted_output(self, mock_googleai_client):
        from model import googleai as provider
        models = await provider.get_models(client=mock_googleai_client)
        ids = [m["id"] for m in models]
        assert ids == sorted(ids)


# ── stream_response ───────────────────────────────────────────────────────

class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_normal_streaming(self, capsys):
        from model import googleai as provider
        ctx = Context("sys prompt")
        ctx.append("user", "Hello")
        ctx.append("assistant", "First reply")
        ctx.append("user", "Follow up")

        chunk1 = MagicMock()
        chunk1.text = "Response "
        chunk2 = MagicMock()
        chunk2.text = "here!"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=AsyncIterator([chunk1, chunk2])
        )

        await provider.stream_response(mock_client, ctx)

        captured = capsys.readouterr()
        assert "Response " in captured.out
        assert "here!" in captured.out
        assert ctx.get_messages()[-1]["content"] == "Response here!"

    @pytest.mark.asyncio
    async def test_multiple_system_messages(self, capsys):
        from model import googleai as provider
        ctx = Context("System 1")
        ctx.messages.append({"role": "system", "content": "System 2"})
        ctx.append("user", "Hello")

        chunk1 = MagicMock()
        chunk1.text = "ok"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=AsyncIterator([chunk1])
        )

        await provider.stream_response(mock_client, ctx)

        call_kwargs = mock_client.aio.models.generate_content_stream.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.system_instruction == "System 1\nSystem 2"

    @pytest.mark.asyncio
    async def test_chunk_without_text(self, capsys):
        from model import googleai as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        chunk1 = MagicMock()
        chunk1.text = None
        chunk2 = MagicMock()
        chunk2.text = "content"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=AsyncIterator([chunk1, chunk2])
        )

        await provider.stream_response(mock_client, ctx)
        assert ctx.get_messages()[-1]["content"] == "content"

    @pytest.mark.asyncio
    async def test_streaming_uses_renderer_when_provided(self):
        from model import googleai as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        chunk1 = MagicMock()
        chunk1.text = "rendered"

        mock_client = MagicMock()
        mock_client.aio.models.generate_content_stream = AsyncMock(
            return_value=AsyncIterator([chunk1])
        )
        renderer = MagicMock()

        await provider.stream_response(mock_client, ctx, renderer=renderer)

        renderer.write_stream.assert_called_once_with("rendered")
        assert ctx.get_messages()[-1]["content"] == "rendered"
