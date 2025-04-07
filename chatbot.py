from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

print("Device set to use CPU")
print("🤖 Start chatting! Type 'exit' to quit.\n")

chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Bye! 👋")
        break

    # Encode user input + eos token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append new input to chat history
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,
    )

    # Decode the response (slice only the new part)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}")


