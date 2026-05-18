"""Unit tests for model/mistral.py — targeting 100% coverage."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from classes.context import Context
from tests.conftest import AsyncIterator


# ── create_client ─────────────────────────────────────────────────────────

class TestCreateClient:
    def test_with_api_key(self):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}):
            with patch("model.mistral.Mistral") as mock_mistral:
                from model import mistral as provider
                client = provider.create_client()
                mock_mistral.assert_called_once_with(api_key="test-key")

    def test_without_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MISTRAL_API_KEY", None)
            from model import mistral as provider
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                provider.create_client()


# ── get_models ────────────────────────────────────────────────────────────

class TestGetModels:
    @pytest.mark.asyncio
    async def test_with_client(self, mock_mistral_client):
        from model import mistral as provider
        models = await provider.get_models(client=mock_mistral_client)
        assert len(models) == 1
        assert models[0]["id"] == "mistral-small-latest"
        assert models[0]["name"] == "Mistral Small"
        assert models[0]["context_window"] == 32000

    @pytest.mark.asyncio
    async def test_without_client_creates_one(self):
        from model import mistral as provider
        mock_client = MagicMock()
        model1 = MagicMock()
        model1.id = "test-model"
        model1.name = "Test"
        model1.description = "desc"
        model1.max_context_length = 8000
        result = MagicMock()
        result.data = [model1]
        mock_client.models.list.return_value = result

        with patch.object(provider, "create_client", return_value=mock_client):
            models = await provider.get_models(client=None)
            assert len(models) == 1

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self):
        from model import mistral as provider
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("API error")

        models = await provider.get_models(client=mock_client)
        assert models == []

    @pytest.mark.asyncio
    async def test_result_without_data_attr(self):
        from model import mistral as provider
        mock_client = MagicMock()
        model1 = MagicMock()
        model1.id = "test-model"
        model1.name = "Test"
        model1.description = "desc"
        model1.max_context_length = 4000
        # Return a plain list (no .data attribute) — hasattr check
        mock_client.models.list.return_value = [model1]

        models = await provider.get_models(client=mock_client)
        assert len(models) == 1

    @pytest.mark.asyncio
    async def test_model_missing_attributes(self):
        from model import mistral as provider
        mock_client = MagicMock()
        model1 = MagicMock(spec=["id"])
        model1.id = "bare-model"
        result = MagicMock()
        result.data = [model1]
        mock_client.models.list.return_value = result

        models = await provider.get_models(client=mock_client)
        assert len(models) == 1
        assert models[0]["id"] == "bare-model"
        assert models[0]["name"] == "bare-model"

    @pytest.mark.asyncio
    async def test_sorted_output(self, mock_mistral_client):
        from model import mistral as provider
        models = await provider.get_models(client=mock_mistral_client)
        ids = [m["id"] for m in models]
        assert ids == sorted(ids)


# ── stream_response ───────────────────────────────────────────────────────

class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_normal_streaming(self, capsys):
        from model import mistral as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        chunk1 = MagicMock()
        chunk1.data.choices = [MagicMock()]
        chunk1.data.choices[0].delta.content = "Hi "

        chunk2 = MagicMock()
        chunk2.data.choices = [MagicMock()]
        chunk2.data.choices[0].delta.content = "there!"

        mock_client = MagicMock()
        mock_client.chat.stream_async = AsyncMock(
            return_value=AsyncIterator([chunk1, chunk2])
        )

        await provider.stream_response(mock_client, ctx)

        captured = capsys.readouterr()
        assert "Hi " in captured.out
        assert "there!" in captured.out
        assert ctx.get_messages()[-1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_delta_none(self, capsys):
        from model import mistral as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        chunk1 = MagicMock()
        chunk1.data.choices = [MagicMock()]
        chunk1.data.choices[0].delta.content = None

        chunk2 = MagicMock()
        chunk2.data.choices = [MagicMock()]
        chunk2.data.choices[0].delta.content = "content"

        mock_client = MagicMock()
        mock_client.chat.stream_async = AsyncMock(
            return_value=AsyncIterator([chunk1, chunk2])
        )

        await provider.stream_response(mock_client, ctx)
        assert ctx.get_messages()[-1]["content"] == "content"

    @pytest.mark.asyncio
    async def test_streaming_uses_renderer_when_provided(self):
        from model import mistral as provider
        ctx = Context("sys")
        ctx.append("user", "Hello")

        chunk1 = MagicMock()
        chunk1.data.choices = [MagicMock()]
        chunk1.data.choices[0].delta.content = "rendered"

        mock_client = MagicMock()
        mock_client.chat.stream_async = AsyncMock(
            return_value=AsyncIterator([chunk1])
        )
        renderer = MagicMock()

        await provider.stream_response(mock_client, ctx, renderer=renderer)

        renderer.write_stream.assert_called_once_with("rendered")
        assert ctx.get_messages()[-1]["content"] == "rendered"
