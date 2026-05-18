import pytest
from unittest.mock import AsyncMock, MagicMock
import classes.cli as cli_module

from classes.cli import ConversationRenderer, CommandCompleter
from classes.context import Context


class TestConversationRenderer:
    def test_status_helpers_and_resume_hint(self, capsys):
        renderer = ConversationRenderer()

        renderer.print_banner("Banner")
        renderer.print_status("Status")
        renderer.print_warning("Warning")
        renderer.print_success("Success")
        renderer.print_error("Error")
        renderer.print_plain("Plain")
        renderer.print_resume_hint("abc-123")

        captured = capsys.readouterr()
        assert "Banner" in captured.out
        assert "Status" in captured.out
        assert "Warning" in captured.out
        assert "Success" in captured.out
        assert "Error" in captured.out
        assert "Plain" in captured.out
        assert "python main.py resume abc-123" in captured.out

    def test_render_message_with_unknown_role_and_prompt(self, capsys):
        renderer = ConversationRenderer()

        prompt = renderer.get_prompt("tool")
        renderer.render_message("tool", "Ran command")

        captured = capsys.readouterr()
        assert "Tool:" in prompt
        assert "Tool:" in captured.out
        assert "Ran command" in captured.out

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
        renderer.end_stream()

        captured = capsys.readouterr()
        assert "AI:" in captured.out
        assert "chunk" in captured.out

    def test_print_model_table_empty_and_values(self, capsys):
        renderer = ConversationRenderer()

        renderer.print_model_table([])
        renderer.print_model_table([
            {"id": "model-a", "context_window": 4096},
            {"id": "model-b", "context_window": None},
        ])

        captured = capsys.readouterr()
        assert "No models found" in captured.out
        assert "model-a" in captured.out
        assert "4096" in captured.out
        assert "model-b" in captured.out
        assert "—" in captured.out


class TestCommandCompleter:
    def test_command_matches(self):
        completer = CommandCompleter({"googleai": MagicMock(), "openai": MagicMock()})
        assert "/model" in completer.get_matches("/m")
        assert "/prune" in completer.get_matches("/")

    def test_get_hint_non_command_root_and_unknown(self):
        completer = CommandCompleter({"googleai": MagicMock()})
        assert completer.get_hint("hello") is None
        assert "Tab completes" in completer.get_hint("/")
        assert completer.get_hint("/wat") is None

    def test_model_matches_from_cache(self):
        completer = CommandCompleter({"googleai": MagicMock(), "openai": MagicMock()})
        completer.model_cache["googleai"] = ["gemini-2.5-pro", "gemma-4"]

        assert completer.get_matches("/model ") == ["googleai", "openai"]
        assert completer.get_matches("/model googleai ") == ["gemini-2.5-pro", "gemma-4"]
        assert completer.get_matches("/model googleai ge") == ["gemini-2.5-pro", "gemma-4"]

    def test_get_matches_misc_branches(self):
        completer = CommandCompleter({"googleai": MagicMock(), "openai": MagicMock()})
        completer.model_cache["googleai"] = ["gemma-4"]

        assert completer.get_matches("hello") == []
        assert completer.get_matches("/prune ") == ["1", "2", "5", "10", "20"]
        assert completer.get_matches("/prune 1") == ["1", "10"]
        assert completer.get_matches("/prune 1 ") == []
        assert completer.get_matches("/model go") == ["googleai"]
        assert completer.get_matches("/model unknown ") == []
        assert completer.get_matches("/other") == []
        assert completer.get_matches("/other ") == []

    def test_hints(self):
        completer = CommandCompleter({"googleai": MagicMock()})
        assert "keep only the last n messages" in completer.get_hint("/prune")
        assert "list or switch models" in completer.get_hint("/model googleai")

    def test_complete_and_display_matches(self, capsys):
        fake_readline = MagicMock()
        fake_readline.get_line_buffer.return_value = "/model "

        original_readline = cli_module.readline
        cli_module.readline = fake_readline
        try:
            completer = CommandCompleter({"googleai": MagicMock(), "openai": MagicMock()})
            assert completer.complete("", 0) == "googleai"
            assert completer.complete("", 1) == "openai"
            assert completer.complete("", 2) is None

            completer.display_matches("", ["googleai", "openai"], 8)
            captured = capsys.readouterr()
            assert "list or switch models" in captured.out
            assert "googleai  openai" in captured.out
            fake_readline.redisplay.assert_called_once()
        finally:
            cli_module.readline = original_readline

    def test_get_matches_fetches_models_lazily(self):
        completer = CommandCompleter({"googleai": MagicMock()})
        completer.ensure_provider_models = MagicMock(
            side_effect=lambda provider_name: completer.model_cache.__setitem__(
                provider_name, ["gemma-4", "gemini-2.5-pro"]
            )
        )

        assert completer.get_matches("/model googleai ") == ["gemma-4", "gemini-2.5-pro"]
        completer.ensure_provider_models.assert_called_once_with("googleai")

    def test_configure_readline_and_complete_without_module(self):
        original_readline = cli_module.readline
        cli_module.readline = None
        try:
            completer = CommandCompleter({"googleai": MagicMock()})
            completer.configure_readline()
            assert completer.complete("", 0) is None
            completer.display_matches("", [], 0)
        finally:
            cli_module.readline = original_readline

    @pytest.mark.asyncio
    async def test_fetch_provider_models_handles_exception(self):
        provider = MagicMock()
        provider.create_client.side_effect = RuntimeError("boom")
        renderer = MagicMock()

        completer = CommandCompleter({"googleai": provider}, renderer=renderer)
        result = await completer.fetch_provider_models("googleai")

        assert result == []
        assert completer.model_cache["googleai"] == []
        assert completer.fetch_errors["googleai"] == "boom"
        renderer.print_warning.assert_called_once()

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
        assert completer.get_cached_models("googleai") == [{"id": "model-a"}, {"id": "model-b"}]

    @pytest.mark.asyncio
    async def test_fetch_provider_models_uses_cache_until_forced(self):
        provider = MagicMock()
        provider.create_client.return_value = MagicMock()
        provider.get_models = AsyncMock(return_value=[{"id": "model-a"}])

        completer = CommandCompleter({"googleai": provider})
        await completer.fetch_provider_models("googleai")
        await completer.fetch_provider_models("googleai")
        await completer.fetch_provider_models("googleai", force=True)

        assert provider.get_models.await_count == 2

    def test_ensure_provider_models_runtime_error_is_ignored(self):
        completer = CommandCompleter({"googleai": MagicMock()})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cli_module.asyncio, "run", MagicMock(side_effect=RuntimeError("loop")))
            completer.ensure_provider_models("googleai")

    def test_ensure_provider_models_noop_when_already_fetched(self):
        completer = CommandCompleter({"googleai": MagicMock()})
        completer.fetch_attempted.add("googleai")

        with pytest.MonkeyPatch.context() as mp:
            run_mock = MagicMock()
            mp.setattr(cli_module.asyncio, "run", run_mock)
            completer.ensure_provider_models("googleai")

        run_mock.assert_not_called()

    def test_configure_readline_with_module(self):
        fake_readline = MagicMock()
        original_readline = cli_module.readline
        cli_module.readline = fake_readline
        try:
            completer = CommandCompleter({"googleai": MagicMock()})
            completer.configure_readline()
            fake_readline.set_completer.assert_called_once()
            fake_readline.set_completer_delims.assert_called_once_with(" \t\n")
            fake_readline.parse_and_bind.assert_called_once_with("tab: complete")
            fake_readline.set_completion_display_matches_hook.assert_called_once()
        finally:
            cli_module.readline = original_readline
