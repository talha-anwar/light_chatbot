from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Create the pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("🤖 Start chatting! Type 'exit' to quit.\n")

chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Bye! 👋")
        break

    # Append new input to chat history
    chat_history += f"{user_input}\n"

    # Generate a response
    input_ids = tokenizer.encode(chat_history, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(f"Bot: {response}")
    chat_history += f"{response}\n"

