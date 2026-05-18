import asyncio

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None


class ConversationRenderer:
    BLUE = "\033[1;34m"
    GREEN = "\033[1;32m"
    MAGENTA = "\033[1;35m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"

    ROLE_LABELS = {
        "system": "System",
        "user": "You",
        "assistant": "AI",
    }

    ROLE_COLORS = {
        "system": BLUE,
        "user": GREEN,
        "assistant": MAGENTA,
    }

    def __init__(self):
        self._stream_open = False

    def print_banner(self, text: str):
        print(f"{self.BLUE}{text}{self.RESET}")

    def print_status(self, text: str):
        print(f"{self.YELLOW}{text}{self.RESET}")

    def print_success(self, text: str):
        print(f"{self.GREEN}{text}{self.RESET}")

    def print_error(self, text: str):
        print(f"{self.RED}{text}{self.RESET}")

    def print_plain(self, text: str = ""):
        print(text)

    def get_prompt(self, role: str = "user") -> str:
        label = self.ROLE_LABELS.get(role, role.capitalize())
        color = self.ROLE_COLORS.get(role, "")
        return f"{color}{label}:{self.RESET} "

    def render_message(self, role: str, content: str):
        label = self.ROLE_LABELS.get(role, role.capitalize())
        color = self.ROLE_COLORS.get(role, "")
        print(f"{color}{label}:{self.RESET} {content}")

    def render_history(self, context):
        for index, message in enumerate(context.get_messages()):
            if index == 0 and message["role"] == "system":
                continue
            self.render_message(message["role"], message["content"])

        if len(context.get_messages()) > 1:
            print()

    def start_stream(self, role: str):
        print(self.get_prompt(role), end="", flush=True)
        self._stream_open = True

    def write_stream(self, chunk: str):
        print(chunk, end="", flush=True)

    def end_stream(self):
        if self._stream_open:
            print()
            self._stream_open = False

    def print_model_table(self, models):
        if not models:
            self.print_status("[No models found]")
            return

        print(f"\n{'ID':<45} {'Context Window':<15}")
        print("-" * 62)
        for model in models:
            ctx_win = str(model["context_window"]) if model["context_window"] else "—"
            print(f"{model['id']:<45} {ctx_win:<15}")
        print()

    def print_resume_hint(self, conversation_id: str):
        print(f"\n{self.YELLOW}To resume this conversation use:{self.RESET}")
        print(f"  python main.py resume {conversation_id}\n")


class CommandCompleter:
    COMMAND_HINTS = {
        "/exit": "/exit: save and quit",
        "/compact": "/compact: summarize older context",
        "/prune": "/prune <n>: keep only the last n messages",
        "/model": "/model <provider> [model_name]: list or switch models",
    }

    def __init__(self, providers):
        self.providers = providers
        self.model_cache = {name: [] for name in providers}
        self.command_names = sorted(self.COMMAND_HINTS)

    async def fetch_provider_models(self, provider_name: str):
        provider = self.providers[provider_name]
        try:
            client = provider.create_client()
            models = await provider.get_models(client)
        except Exception:
            models = []

        self.model_cache[provider_name] = sorted(model["id"] for model in models)
        return self.model_cache[provider_name]

    async def fetch_all_models(self):
        await asyncio.gather(
            *(self.fetch_provider_models(provider_name) for provider_name in self.providers)
        )

    def configure_readline(self):
        if readline is None:  # pragma: no cover
            return

        readline.set_completer(self.complete)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
        readline.set_completion_display_matches_hook(self.display_matches)

    def get_hint(self, line_buffer: str):
        stripped = line_buffer.strip()
        if not stripped.startswith("/"):
            return None

        if stripped == "/":
            return "Tab completes: /exit, /compact, /prune, /model"
        if stripped.startswith("/prune"):
            return self.COMMAND_HINTS["/prune"]
        if stripped.startswith("/model"):
            return self.COMMAND_HINTS["/model"]

        return self.COMMAND_HINTS.get(stripped.split()[0])

    def get_matches(self, line_buffer: str):
        stripped = line_buffer.lstrip()
        if not stripped.startswith("/"):
            return []

        parts = stripped.split()
        ends_with_space = line_buffer.endswith(" ")

        if not parts:
            return self.command_names

        if len(parts) == 1 and not ends_with_space:
            return [cmd for cmd in self.command_names if cmd.startswith(parts[0])]

        command = parts[0]
        if command == "/prune":
            if len(parts) == 1 and ends_with_space:
                return ["1", "2", "5", "10", "20"]
            if len(parts) == 2 and not ends_with_space:
                return [n for n in ["1", "2", "5", "10", "20"] if n.startswith(parts[1])]
            return []

        if command == "/model":
            provider_names = sorted(self.providers)
            if len(parts) == 1 and ends_with_space:
                return provider_names
            if len(parts) == 2 and not ends_with_space:
                return [name for name in provider_names if name.startswith(parts[1])]
            if len(parts) >= 2:
                provider_name = parts[1]
                model_prefix = ""
                if len(parts) >= 3 and not ends_with_space:
                    model_prefix = parts[2]
                if provider_name in self.model_cache:
                    return [
                        model_id
                        for model_id in self.model_cache[provider_name]
                        if model_id.startswith(model_prefix)
                    ]
            return []

        return []

    def complete(self, text, state):
        if readline is None:  # pragma: no cover
            return None

        matches = self.get_matches(readline.get_line_buffer())
        if state < len(matches):
            return matches[state]
        return None

    def display_matches(self, substitution, matches, longest_match_length):
        if readline is None:  # pragma: no cover
            return

        line_buffer = readline.get_line_buffer()
        hint = self.get_hint(line_buffer)
        print()
        if hint:
            print(hint)
        if matches:
            print("  ".join(matches))
        readline.redisplay()
