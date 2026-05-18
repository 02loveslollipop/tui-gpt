# yacpt (Yet another CLI for Pretrained Transformers)

A Terminal User Interface (TUI) / CLI for interacting with various Large Language Models (LLMs). This project is a modular, async CLI chat application that supports multiple model providers and features automatic context management.

## Features

*   **Multi-Provider Support:** Switch between different LLM providers seamlessly. Currently supports:
    *   OpenAI
    *   Google AI (Gemini)
    *   Mistral AI
*   **Dynamic Model Switching:** Change the active provider and model directly from the chat interface.
*   **Context Management:**
    *   **Auto-Compaction:** Automatically compacts the conversation history when the token limit is reached.
    *   **Manual Pruning:** Commands to manually prune the conversation history to keep only the last *n* messages.
*   **Fun Persona:** By default, the AI adopts a cheerful, chaotic rabbit-themed mascot persona called "pekofy.ai", obsessed with glasses.

## Requirements

*   Python 3.8+
*   Dependencies listed in `requirements.txt`:
    *   `openai`
    *   `python-dotenv`
    *   `tiktoken`
    *   `google-genai`
    *   `mistralai`

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd yacpt
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    Create a `.env` file in the root directory and add your API keys for the providers you intend to use.
    ```env
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_API_KEY=your_google_api_key
    MISTRAL_API_KEY=your_mistral_api_key
    ```

## Usage

Run the main script to start the CLI:

```bash
python main.py
```

### Commands

Within the chat interface, you can use the following commands:

*   `/exit` or `/quit`: Exit the application.
*   `/compact`: Manually trigger context compaction to summarize previous messages.
*   `/prune <n>`: Prune the context history to keep only the last `<n>` messages.
*   `/model <provider> [custom_model_name]`: Switch the active model provider. 
    *   Example: `/model mistral`
    *   Example with custom model: `/model openai gpt-4o`

## Project Structure

*   `main.py`: The entry point and main chat loop.
*   `classes/context.py`: Handles token counting, context tracking, and compaction logic.
*   `model/`: Contains provider-specific integration modules (`openai.py`, `googleai.py`, `mistral.py`). Each module defines how to create a client and stream responses for that specific provider.
