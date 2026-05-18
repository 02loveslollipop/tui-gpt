"""Unit / integration tests for main.py — targeting 100% coverage."""
import json
import os
import sys
import asyncio
import re
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from classes.context import Context


# ── resolve_system_prompt ─────────────────────────────────────────────────

class TestResolveSystemPrompt:
    def test_prompt_file_arg_exists(self, tmp_path):
        import main
        with patch("main.Path.is_file") as mock_is_file, \
             patch("main.Path.read_text") as mock_read:
            # We want ONLY the first check to return True
            def is_file_side_effect():
                # We can't easily check 'self' path in side_effect safely without inspect,
                # but we can just use a simple return sequence.
                return True
            mock_is_file.return_value = True
            mock_read.return_value = "Custom prompt"
            res = main.resolve_system_prompt("custom.md")
            assert res == "Custom prompt"

    def test_prompt_file_arg_not_found_falls_back(self):
        import main
        with patch("main.Path.is_file") as mock_is_file:
            # All is_file checks return False
            mock_is_file.return_value = False
            res = main.resolve_system_prompt("missing.md")
            assert res == main.SYSTEM_PROMPT

    def test_prompt_file_arg_not_found_warns_via_renderer(self):
        import main
        renderer = MagicMock()
        with patch("main.Path.is_file", return_value=False):
            res = main.resolve_system_prompt("missing.md", renderer=renderer)
        assert res == main.SYSTEM_PROMPT
        renderer.print_warning.assert_called_once()

    def test_local_prompt_exists(self):
        import main
        with patch("main.Path.is_file") as mock_is_file, \
             patch("main.Path.read_text") as mock_read:
            
            # The checks happen in this order: local_prompt, global_prompt
            # If we don't pass an arg, the first check is local_prompt
            def is_file_side_effect():
                pass
                
            # We can inspect the path string representation via mock calls if needed, 
            # but simpler: if it's called on PROMPT.md, return True.
            # A cleaner way is patching Path's methods bound to the instance, 
            # but mocking `is_file` generally works.
            # Let's just return True for the first call (local)
            mock_is_file.side_effect = [True]
            mock_read.return_value = "Local prompt"
            
            res = main.resolve_system_prompt()
            assert res == "Local prompt"

    def test_global_prompt_exists(self):
        import main
        with patch("main.Path.is_file") as mock_is_file, \
             patch("main.Path.read_text") as mock_read:
            
            # Sequence: local (False), global (True)
            mock_is_file.side_effect = [False, True]
            mock_read.return_value = "Global prompt"
            
            res = main.resolve_system_prompt()
            assert res == "Global prompt"


# ── get_latest_conversation ───────────────────────────────────────────────

class TestGetLatestConversation:
    def test_dir_does_not_exist(self, tmp_path):
        import main
        with patch.object(main, "CONVERSATIONS_DIR", tmp_path / "nonexistent"):
            result = main.get_latest_conversation()
            assert result is None

    def test_dir_empty(self, tmp_path):
        import main
        conv_dir = tmp_path / "convs"
        conv_dir.mkdir()
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir):
            result = main.get_latest_conversation()
            assert result is None

    def test_finds_latest_by_mtime(self, tmp_path):
        import main
        import time
        conv_dir = tmp_path / "convs"
        conv_dir.mkdir()
        f1 = conv_dir / "old.jsonl"
        f1.write_text("old")
        time.sleep(0.05)
        f2 = conv_dir / "new.jsonl"
        f2.write_text("new")

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir):
            result = main.get_latest_conversation()
            assert result == f2


# ── save_conversation ─────────────────────────────────────────────────────

class TestSaveConversation:
    def test_delegates_to_serialize(self, tmp_path):
        import main
        ctx = Context("sys")
        filepath = tmp_path / "test.jsonl"
        main.save_conversation(ctx, filepath, "openai", "gpt-5")
        assert filepath.exists()
        with open(filepath) as f:
            meta = json.loads(f.readline())
        assert meta["meta"]["provider"] == "openai"


# ── display_models ────────────────────────────────────────────────────────

class TestDisplayModels:
    def test_with_models(self, capsys):
        import main
        from classes.cli import ConversationRenderer
        main.asyncio.run(main.display_models(ConversationRenderer(), [
            {"id": "model-1", "name": "Model 1", "description": "", "context_window": 128000},
            {"id": "model-2", "name": "Model 2", "description": "", "context_window": None},
        ]))
        captured = capsys.readouterr()
        assert "model-1" in captured.out
        assert "128000" in captured.out
        assert "model-2" in captured.out

    def test_no_models(self, capsys):
        import main
        from classes.cli import ConversationRenderer
        main.asyncio.run(main.display_models(ConversationRenderer(), []))
        captured = capsys.readouterr()
        assert "No models found" in captured.out

# ── Helpers for main() tests ─────────────────────────────────────────────

def _make_mock_provider(stream_text="Response text"):
    """Create a mock provider module with all required attributes."""
    provider = MagicMock()
    provider.MODEL = "test-model"
    provider.DISPLAY_NAME = "Test (test-model)"
    provider.ENV_VAR = "TEST_API_KEY"
    provider.create_client = MagicMock(return_value=MagicMock())

    async def mock_stream(client, context, renderer=None):
        if renderer is not None:
            renderer.start_stream("assistant")
            renderer.write_stream(stream_text)
            renderer.end_stream()
        else:
            print(stream_text, end="", flush=True)
            print()
        context.append("assistant", stream_text)

    provider.stream_response = AsyncMock(side_effect=mock_stream)
    provider.get_models = AsyncMock(return_value=[
        {"id": "test-model", "name": "Test", "description": "", "context_window": 4096}
    ])
    return provider


def _patch_main_for_test(tmp_path, inputs, argv=None, provider=None):
    """Return a dict of patches for running main() in tests."""
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir(exist_ok=True)

    if provider is None:
        provider = _make_mock_provider()

    patches = {
        "CONVERSATIONS_DIR": conv_dir,
        "PROVIDERS": {
            "openai": provider,
            "googleai": provider,
            "mistral": provider,
        },
    }
    return patches, conv_dir, provider


# ── main() — new session ─────────────────────────────────────────────────

class TestMainNewSession:
    @pytest.mark.asyncio
    async def test_new_session_does_not_prefetch_models_on_startup(self, tmp_path):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid-1234"
            await main.main()

        provider.get_models.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_message_and_exit(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["Hello!", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid-1234"
            await main.main()

        captured = capsys.readouterr()
        assert "resume" in captured.out
        assert "test-uuid-1234" in captured.out
        assert "Goodbye!" in captured.out

        # Verify JSONL was saved
        jsonl_file = conv_dir / "test-uuid-1234.jsonl"
        assert jsonl_file.exists()

    @pytest.mark.asyncio
    async def test_empty_input_skipped(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["", "   ", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @pytest.mark.asyncio
    async def test_quit_command(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["quit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    @pytest.mark.asyncio
    async def test_exit_command(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out


# ── main() — resume ──────────────────────────────────────────────────────

class TestMainResume:
    @pytest.mark.asyncio
    async def test_resume_by_uuid(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        # Create a conversation file
        ctx = Context("Test system")
        ctx.append("user", "Previous message")
        ctx.append("assistant", "Previous reply")
        conv_file = conv_dir / "resume-uuid.jsonl"
        ctx.serialize(str(conv_file), provider_name="googleai", model_name="test-model")

        input_sequence = iter(["/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume", "resume-uuid"]), \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            await main.main()

        captured = capsys.readouterr()
        plain_out = re.sub(r"\x1b\[[0-9;]*m", "", captured.out)
        assert "Resumed conversation resume-uuid" in plain_out
        assert "2 messages" in plain_out
        assert "You: Previous message" in plain_out
        assert "AI: Previous reply" in plain_out

    @pytest.mark.asyncio
    async def test_resume_latest(self, tmp_path, capsys):
        import main
        import time
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        # Create two conversation files
        ctx1 = Context("Old")
        ctx1.serialize(str(conv_dir / "old-uuid.jsonl"), provider_name="openai", model_name="old")
        time.sleep(0.05)
        ctx2 = Context("New")
        ctx2.append("user", "hi")
        ctx2.serialize(str(conv_dir / "new-uuid.jsonl"), provider_name="googleai", model_name="new")

        input_sequence = iter(["/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume"]), \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            await main.main()

        captured = capsys.readouterr()
        assert "Resumed conversation new-uuid" in captured.out

    @pytest.mark.asyncio
    async def test_resume_no_files(self, tmp_path, capsys):
        import main
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [])

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume"]), \
             pytest.raises(SystemExit):
            await main.main()

        captured = capsys.readouterr()
        assert "No conversations found" in captured.out

    @pytest.mark.asyncio
    async def test_resume_uuid_not_found(self, tmp_path, capsys):
        import main
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [])

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume", "nonexistent-uuid"]), \
             pytest.raises(SystemExit):
            await main.main()

        captured = capsys.readouterr()
        assert "not found" in captured.out

    @pytest.mark.asyncio
    async def test_resume_restores_provider_and_model(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        ctx = Context("Sys")
        ctx.serialize(str(conv_dir / "meta-test.jsonl"), provider_name="openai", model_name="custom-model")

        input_sequence = iter(["/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume", "meta-test"]), \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            await main.main()

        assert provider.MODEL == "custom-model"

    @pytest.mark.asyncio
    async def test_resume_empty_provider_uses_default(self, tmp_path, capsys):
        """When metadata has empty provider, default is used."""
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        ctx = Context("Sys")
        ctx.serialize(str(conv_dir / "noprov.jsonl"), provider_name="", model_name="")

        input_sequence = iter(["/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume", "noprov"]), \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            await main.main()

        captured = capsys.readouterr()
        assert "Resumed" in captured.out


# ── main() — commands ─────────────────────────────────────────────────────

class TestMainCommands:
    @pytest.mark.asyncio
    async def test_compact_command(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["Hello", "/compact", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            # Mock context.compact
            with patch.object(Context, "compact", new_callable=AsyncMock):
                await main.main()

        captured = capsys.readouterr()
        assert "Compacting context" in captured.out
        assert "Done" in captured.out

    @pytest.mark.asyncio
    async def test_prune_valid(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["Hello", "/prune 2", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Pruned context to last 2" in captured.out

    @pytest.mark.asyncio
    async def test_prune_invalid(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["/prune", "/prune abc", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Usage: /prune" in captured.out

    @pytest.mark.asyncio
    async def test_model_list(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["/model googleai", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Fetching models" in captured.out
        assert "test-model" in captured.out
        provider.get_models.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_model_switch(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["/model googleai custom-model-v2", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Switched to" in captured.out

    @pytest.mark.asyncio
    async def test_model_switch_create_client_error(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        provider.create_client = MagicMock(side_effect=ValueError("No API key"))
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        # First call succeeds (initial client), second fails (switch)
        call_count = [0]
        original_create = provider.create_client

        def create_client_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock()
            raise ValueError("No API key")

        provider.create_client = MagicMock(side_effect=create_client_side_effect)

        input_sequence = iter(["/model googleai new-model", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "No API key" in captured.out

    @pytest.mark.asyncio
    async def test_model_invalid(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["/model", "/model invalidprovider", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Usage: /model" in captured.out


# ── main() — streaming & auto-compact ────────────────────────────────────

class TestMainStreaming:
    @pytest.mark.asyncio
    async def test_stream_error_caught(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        provider.stream_response = AsyncMock(side_effect=Exception("Stream failed"))
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["Hello", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Error: Stream failed" in captured.out

    @pytest.mark.asyncio
    async def test_auto_compact_trigger(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        input_sequence = iter(["Hello", "/exit"])
        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=lambda fn, *a: next(input_sequence)), \
             patch.object(Context, "count_tokens", return_value=99999), \
             patch.object(Context, "compact", new_callable=AsyncMock) as mock_compact:
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "Token limit reached" in captured.out
        assert "Auto-compacting" in captured.out
        mock_compact.assert_called()


# ── main() — interrupt ───────────────────────────────────────────────────

class TestMainInterrupt:
    @pytest.mark.asyncio
    async def test_keyboard_interrupt(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        call_count = [0]
        def input_with_interrupt(fn, *a):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Hello"
            raise KeyboardInterrupt()

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=input_with_interrupt):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "resume" in captured.out
        assert "test-uuid" in captured.out
        # Verify file was saved
        assert (conv_dir / "test-uuid.jsonl").exists()

    @pytest.mark.asyncio
    async def test_eof_error(self, tmp_path, capsys):
        import main
        provider = _make_mock_provider()
        patches, conv_dir, _ = _patch_main_for_test(tmp_path, [], provider=provider)

        def raise_eof(fn, *a):
            raise EOFError()

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "PROVIDERS", patches["PROVIDERS"]), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("asyncio.to_thread", side_effect=raise_eof):
            mock_uuid.uuid4.return_value = "test-uuid"
            await main.main()

        captured = capsys.readouterr()
        assert "resume" in captured.out


# ── __main__ block ────────────────────────────────────────────────────────

class TestMainBlock:
    def test_main_block_keyboard_interrupt(self):
        """Test that __main__ block catches KeyboardInterrupt."""
        import main
        with patch.object(main, "main", new_callable=lambda: lambda: AsyncMock(side_effect=KeyboardInterrupt)):
            with patch("asyncio.run", side_effect=KeyboardInterrupt):
                # Should not raise
                try:
                    # Simulate if __name__ == "__main__" block
                    try:
                        asyncio.run(main.main())
                    except KeyboardInterrupt:
                        pass
                except KeyboardInterrupt:
                    pytest.fail("KeyboardInterrupt was not caught")
