import torch
import torch.nn.functional as F
import tiktoken
from gpt2 import GPT, GPTConfig

def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=40, device="cpu"):
    # Initialize the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    
    # Convert to tensor and add batch dimension: shape (1, T)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Autoregressive generation loop
    with torch.no_grad(): # Disables gradient tracking to save memory and speed up inference
        for _ in range(max_new_tokens):
            # Crop the context to the model's max block size
            idx_cond = idx[:, -model.config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            
            # Pluck the logits for the final step and scale by temperature
            # logits shape is (B, T, vocab_size) -> we want (B, vocab_size)
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k most likely options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to normalized probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
    # Decode the tensor back to strings
    generated_tokens = idx[0].tolist()
    return enc.decode(generated_tokens)

def main():
    # Load the model
    print("Loading the model...")

    # Use CUDA if available, otherwise fallback to
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        config = GPTConfig() # Use default GPT-2 configuration
        model = GPT(config).to(device)
        model_save_path = "model.pth"
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval() # Put model into evaluation mode
    except FileNotFoundError:
        print(f"Model weights file '{model_save_path}' not found. Please make sure the model is trained and saved properly.")
        return
    except Exception as e:
        print(f"An error occured while loading the model. Please make sure the model is trained and saved properly.")
        print(f"Error trace: {e}")
        return

    print("Model loaded successfully.")

    while True:
        print()
        print("-------- Generation --------")
        print("Please write the prompt for generation, or 'q' to quit the program.")
        input_text = input("Input:")

        if (input_text == "q"):
            print("Exiting...")
            break

        print(f"Generating output...")
        output = generate(model=model, prompt=input_text, device=device)
        print(f"Output: {output}")

if __name__ == "__main__":
    main()