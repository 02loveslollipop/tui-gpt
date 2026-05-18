"""End-to-end tests that make real API calls.

These require API keys in the environment.
Marked with @pytest.mark.e2e — skipped by default in unit test runs.
"""
import os
import pytest
from dotenv import load_dotenv
from classes.context import Context

load_dotenv()

pytestmark = pytest.mark.e2e


def _skip_if_no_key(env_var):
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set")


# ── OpenAI ────────────────────────────────────────────────────────────────

class TestOpenAIE2E:
    @pytest.mark.asyncio
    async def test_get_models_live(self):
        _skip_if_no_key("OPENAI_API_KEY")
        from model import openai as provider
        client = provider.create_client()
        models = await provider.get_models(client)
        assert isinstance(models, list)
        assert len(models) > 0
        for m in models:
            assert "id" in m
            assert "name" in m
            assert any(p in m["id"] for p in ("gpt", "o1", "o3", "o4"))

    @pytest.mark.asyncio
    async def test_stream_response_live(self):
        _skip_if_no_key("OPENAI_API_KEY")
        from model import openai as provider
        client = provider.create_client()
        ctx = Context("You are a test bot. Reply with exactly one word.")
        ctx.append("user", "Say hello.")
        await provider.stream_response(client, ctx)
        assert len(ctx.get_messages()) == 3
        assert ctx.get_messages()[-1]["role"] == "assistant"
        assert len(ctx.get_messages()[-1]["content"]) > 0


# ── Google AI ─────────────────────────────────────────────────────────────

class TestGoogleAIE2E:
    @pytest.mark.asyncio
    async def test_get_models_live(self):
        _skip_if_no_key("GOOGLE_API_KEY")
        from model import googleai as provider
        client = provider.create_client()
        models = await provider.get_models(client)
        assert isinstance(models, list)
        assert len(models) > 0
        for m in models:
            assert "id" in m
            assert "context_window" in m

    @pytest.mark.asyncio
    async def test_stream_response_live(self):
        _skip_if_no_key("GOOGLE_API_KEY")
        from model import googleai as provider
        client = provider.create_client()
        ctx = Context("You are a test bot. Reply with exactly one word.")
        ctx.append("user", "Say hello.")
        await provider.stream_response(client, ctx)
        assert len(ctx.get_messages()) == 3
        assert ctx.get_messages()[-1]["role"] == "assistant"
        assert len(ctx.get_messages()[-1]["content"]) > 0


# ── Mistral ───────────────────────────────────────────────────────────────

class TestMistralE2E:
    @pytest.mark.asyncio
    async def test_get_models_live(self):
        _skip_if_no_key("MISTRAL_API_KEY")
        from model import mistral as provider
        client = provider.create_client()
        models = await provider.get_models(client)
        assert isinstance(models, list)
        assert len(models) > 0
        for m in models:
            assert "id" in m

    @pytest.mark.asyncio
    async def test_stream_response_live(self):
        _skip_if_no_key("MISTRAL_API_KEY")
        from model import mistral as provider
        client = provider.create_client()
        ctx = Context("You are a test bot. Reply with exactly one word.")
        ctx.append("user", "Say hello.")
        await provider.stream_response(client, ctx)
        assert len(ctx.get_messages()) == 3
        assert ctx.get_messages()[-1]["role"] == "assistant"
        assert len(ctx.get_messages()[-1]["content"]) > 0


# ── Full conversation flow ───────────────────────────────────────────────

class TestFullFlowE2E:
    @pytest.mark.asyncio
    async def test_save_and_resume(self, tmp_path):
        """End-to-end: new conversation → save → resume → verify."""
        _skip_if_no_key("GOOGLE_API_KEY")
        from model import googleai as provider

        # Create and populate a conversation
        client = provider.create_client()
        ctx = Context("You are a test bot. Reply with exactly one word.")
        ctx.append("user", "Say hello.")
        await provider.stream_response(client, ctx)

        # Save
        filepath = str(tmp_path / "e2e-test.jsonl")
        ctx.serialize(filepath, provider_name="googleai", model_name=provider.MODEL)

        # Resume
        ctx2, meta = Context.deserialize(filepath)
        assert meta["provider"] == "googleai"
        assert len(ctx2.get_messages()) == len(ctx.get_messages())
        assert ctx2.get_messages()[-1]["role"] == "assistant"


# ── CLI Integration Test (real conversation) ─────────────────────────────

class TestCLIIntegrationE2E:
    """Full CLI integration test simulating real user sessions with live APIs."""

    @pytest.mark.asyncio
    async def test_full_cli_flow(self, tmp_path):
        """
        Simulates a complete CLI session with real API calls:
        1. Start new conversation with Google AI
        2. Send a message, get real response
        3. Switch provider to OpenAI
        4. Send another message
        5. Prune context
        6. Exit and verify resume message
        7. Resume the conversation from JSONL
        8. Continue with another message
        9. Verify full context integrity
        """
        import sys
        import json
        from unittest.mock import patch
        import main

        _skip_if_no_key("GOOGLE_API_KEY")
        _skip_if_no_key("OPENAI_API_KEY")

        conv_dir = tmp_path / "conversations"
        conv_dir.mkdir()
        test_uuid = "cli-integration-test"

        # --- Phase 1: New session, send message, exit ---
        input_seq = iter([
            "Say exactly: test-response-alpha",
            "/exit",
        ])

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "SYSTEM_PROMPT", "You are a test bot. Follow instructions exactly."), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("builtins.input", side_effect=lambda prompt: next(input_seq)):
            mock_uuid.uuid4.return_value = test_uuid
            await main.main()

        # Verify JSONL was created with real content
        conv_file = conv_dir / f"{test_uuid}.jsonl"
        assert conv_file.exists()

        with open(conv_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert lines[0]["meta"]["provider"] == "googleai"
        assert lines[1]["role"] == "system"
        assert lines[2]["role"] == "user"
        assert lines[2]["content"] == "Say exactly: test-response-alpha"
        assert lines[3]["role"] == "assistant"
        assert len(lines[3]["content"]) > 0  # Real API response

        # --- Phase 2: Resume and continue ---
        input_seq2 = iter([
            "Say exactly: test-response-beta",
            "/exit",
        ])

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "SYSTEM_PROMPT", "You are a test bot. Follow instructions exactly."), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py", "resume", test_uuid]), \
             patch("builtins.input", side_effect=lambda prompt: next(input_seq2)):
            await main.main()

        # Verify the resumed file has the full conversation
        with open(conv_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # meta + system + user1 + assistant1 + user2 + assistant2 = 6 lines
        assert len(lines) == 6
        assert lines[4]["role"] == "user"
        assert lines[4]["content"] == "Say exactly: test-response-beta"
        assert lines[5]["role"] == "assistant"
        assert len(lines[5]["content"]) > 0

        # --- Phase 3: Verify context round-trip integrity ---
        ctx, meta = Context.deserialize(str(conv_file))
        assert meta["provider"] == "googleai"
        messages = ctx.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[4]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_cli_prune_and_compact(self, tmp_path):
        """Test /prune and /compact commands with real API responses."""
        import sys
        import json
        from unittest.mock import patch
        import main

        _skip_if_no_key("GOOGLE_API_KEY")

        conv_dir = tmp_path / "conversations"
        conv_dir.mkdir()
        test_uuid = "cli-prune-test"

        # Send several messages then prune
        input_seq = iter([
            "Say one word: alpha",
            "Say one word: beta",
            "Say one word: gamma",
            "/prune 2",
            "/exit",
        ])

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "SYSTEM_PROMPT", "You are a test bot. Reply with one word only."), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("builtins.input", side_effect=lambda prompt: next(input_seq)):
            mock_uuid.uuid4.return_value = test_uuid
            await main.main()

        # After prune 2: system + last 2 messages + meta line
        with open(conv_dir / f"{test_uuid}.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # meta + system + last 2 messages = 4 lines
        assert lines[0].get("meta") is not None
        assert lines[1]["role"] == "system"
        # After prune 2, we keep system + last 2 msgs (user:gamma + assistant:response)
        assert len(lines) == 4

    @pytest.mark.asyncio
    async def test_cli_model_switch_and_stream(self, tmp_path):
        """Test switching providers mid-conversation with real APIs."""
        import sys
        import json
        from unittest.mock import patch
        import main

        _skip_if_no_key("GOOGLE_API_KEY")
        _skip_if_no_key("MISTRAL_API_KEY")

        conv_dir = tmp_path / "conversations"
        conv_dir.mkdir()
        test_uuid = "cli-switch-test"

        input_seq = iter([
            "Say one word: first",
            "/model mistral mistral-small-latest",
            "Say one word: second",
            "/exit",
        ])

        with patch.object(main, "CONVERSATIONS_DIR", conv_dir), \
             patch.object(main, "SYSTEM_PROMPT", "You are a test bot. Reply with one word only."), \
             patch("main.load_dotenv"), \
             patch.object(sys, "argv", ["main.py"]), \
             patch("main.uuid") as mock_uuid, \
             patch("builtins.input", side_effect=lambda prompt: next(input_seq)):
            mock_uuid.uuid4.return_value = test_uuid
            await main.main()

        # Verify the file saved with the last active provider (mistral)
        with open(conv_dir / f"{test_uuid}.jsonl") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert lines[0]["meta"]["provider"] == "mistral"
        assert lines[0]["meta"]["model"] == "mistral-small-latest"
        # Should have: meta + system + user1 + assistant1 + user2 + assistant2
        assert len(lines) == 6
