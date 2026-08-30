[README.md](https://github.com/user-attachments/files/31608887/README.md)
# Project 1: Rule-Based AI Chatbot
**DecodeLabs Industrial Training Kit — Batch 2026**

## What this is
A rule-based chatbot built in Python that matches user input to responses
using an O(1) dictionary lookup instead of a linear if-elif ladder.

## Files
- `rule_based_chatbot.py` — the complete, runnable chatbot script

## How to run
1. Make sure Python 3.6+ is installed (`python3 --version`)
2. Open a terminal in this folder
3. Run:
   ```
   python3 rule_based_chatbot.py
   ```
   (On Windows, use `python rule_based_chatbot.py`)
4. Chat with the bot. Type `exit` or `quit` to end the session.

## Requirements met
- **Input Loop** — continuous `while True` loop with a clean exit strategy
- **Sanitization** — all input normalized via `.lower().strip()`
- **Efficient Logic Architecture** — dictionary/hash-map lookup (O(1)),
  no if-elif ladder
- **Atomic fallback handling** — `.get(key, default)` used for lookup +
  default response in a single operation
- **Knowledge Base** — 7 unique intent/response pairs (exceeds the
  5-intent minimum), covering greetings, status checks, identity, help,
  and capabilities
- **Graceful unknown-input handling** — default fallback response
- **Modular structure** — `sanitize_input()`, `get_response()`,
  `run_chatbot()`
- **IPO model** — `run_chatbot()` follows Input (`input()`) → Process
  (`sanitize_input()`, `get_response()`) → Output (`print()`) on every
  loop iteration

## Sample run
```
DecodeBot: Hello! I'm your rule-based assistant. Type 'exit' or 'quit' to end our chat.

You: Hi
DecodeBot: Hello there! Welcome to DecodeLabs Support Bot.

You:   HELLO  
DecodeBot: Hi! How can I assist you today?

You: How Are You
DecodeBot: I'm just a bunch of if-free dictionary lookups, but I'm running smoothly!

You: asdkjaskjd
DecodeBot: I'm sorry, I don't understand that yet. Type 'help' to see what I can do.

You: EXIT
DecodeBot: Goodbye! Thanks for chatting. 👋
```

## Author
Submitted as part of DecodeLabs Project 1 — Industrial Training Kit,
Batch 2026.
