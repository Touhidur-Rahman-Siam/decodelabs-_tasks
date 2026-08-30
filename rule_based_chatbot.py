KNOWLEDGE_BASE = {
    "hi": "Hello there! Welcome to DecodeLabs Support Bot.",
    "hello": "Hi! How can I assist you today?",
    "how are you": "I'm just a bunch of if-free dictionary lookups, but I'm running smoothly!",
    "what is your name": "I'm DecodeBot, a rule-based assistant built at DecodeLabs.",
    "help": "Sure — you can ask me about my name, how I'm doing, or what I can do.",
    "what can you do": "I can respond to a fixed set of known intents using instant dictionary lookups.",
    "bye": "Goodbye! Have a great day.",
}

EXIT_COMMANDS = {"exit", "quit"}
DEFAULT_RESPONSE = "I'm sorry, I don't understand that yet. Type 'help' to see what I can do."


def sanitize_input(raw_input: str) -> str:
    return raw_input.lower().strip()


def get_response(user_intent: str) -> str:
    return KNOWLEDGE_BASE.get(user_intent, DEFAULT_RESPONSE)


def run_chatbot() -> None:
    print("DecodeBot: Hello! I'm your rule-based assistant. Type 'exit' or 'quit' to end our chat.\n")

    while True:
        raw_input = input("You: ")
        clean_input = sanitize_input(raw_input)

        if clean_input in EXIT_COMMANDS:
            print("DecodeBot: Goodbye! Thanks for chatting. \U0001F44B")
            break

        response = get_response(clean_input)
        print(f"DecodeBot: {response}")


if __name__ == "__main__":
    run_chatbot()
