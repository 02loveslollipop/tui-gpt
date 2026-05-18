import pytest
from unittest.mock import AsyncMock, MagicMock

from classes.cli import ConversationRenderer, CommandCompleter
from classes.context import Context


class TestConversationRenderer:
    def test_render_history_uses_role_labels(self, capsys):
        renderer = ConversationRenderer()
        ctx = Context("System prompt")
        ctx.append("user", "Hello")
        ctx.append("assistant", "Hi there")

        renderer.render_history(ctx)

        captured = capsys.readouterr()
        assert "You:" in captured.out
        assert "Hello" in captured.out
        assert "AI:" in captured.out
        assert "Hi there" in captured.out
        assert "System prompt" not in captured.out

    def test_stream_lifecycle(self, capsys):
        renderer = ConversationRenderer()

        renderer.start_stream("assistant")
        renderer.write_stream("chunk")
        renderer.end_stream()

        captured = capsys.readouterr()
        assert "AI:" in captured.out
        assert "chunk" in captured.out


class TestCommandCompleter:
    def test_command_matches(self):
        completer = CommandCompleter({"googleai": MagicMock(), "openai": MagicMock()})
        assert "/model" in completer.get_matches("/m")
        assert "/prune" in completer.get_matches("/")

    def test_model_matches_from_cache(self):
        completer = CommandCompleter({"googleai": MagicMock(), "openai": MagicMock()})
        completer.model_cache["googleai"] = ["gemini-2.5-pro", "gemma-4"]

        assert completer.get_matches("/model ") == ["googleai", "openai"]
        assert completer.get_matches("/model googleai ") == ["gemini-2.5-pro", "gemma-4"]
        assert completer.get_matches("/model googleai ge") == ["gemini-2.5-pro", "gemma-4"]

    def test_hints(self):
        completer = CommandCompleter({"googleai": MagicMock()})
        assert "keep only the last n messages" in completer.get_hint("/prune")
        assert "list or switch models" in completer.get_hint("/model googleai")

    @pytest.mark.asyncio
    async def test_fetch_all_models_populates_cache(self):
        provider = MagicMock()
        provider.create_client.return_value = MagicMock()
        provider.get_models = AsyncMock(return_value=[
            {"id": "model-b"},
            {"id": "model-a"},
        ])

        completer = CommandCompleter({"googleai": provider})
        await completer.fetch_all_models()

        assert completer.model_cache["googleai"] == ["model-a", "model-b"]
