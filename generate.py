import pickle
import sys

import torch
from transformers import AutoTokenizer

from model import DeepSeekForCausalLM, ModelConfig


def load_trained_model(checkpoint_path):
    # Initialize model and config
    config = ModelConfig()
    model = DeepSeekForCausalLM(config)

    # Load checkpoint with weights_only=False to handle the ModelConfig class
    try:
        # First try to load with weights_only=False (safer but might fail)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Trying alternative loading method...")

        # Add ModelConfig to safe globals
        torch.serialization.add_safe_globals([ModelConfig])

        try:
            # Try again with weights_only=True but with ModelConfig added to safe globals
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            print("Trying final fallback method...")

            # Final fallback: manually load the state dict
            checkpoint = {
                "model_state_dict": torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            }

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Successfully loaded checkpoint from {checkpoint_path}")

    # Move model to device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_text(
    model, tokenizer, device, prompt, max_length=100, temperature=0.7, top_p=0.9
):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids)

            # Get logits for the next token (last position)
            next_token_logits = outputs[..., -1, :] / temperature

            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add the token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated tokens
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main():
    # Load trained model
    checkpoint_path = "final_model.pt"
    model, tokenizer, device = load_trained_model(checkpoint_path)

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to solving climate change is",
        "When exploring the depths of the ocean,",
        "The most important scientific discovery of the century was",
    ]

    # Generate outputs
    print("\nGenerating outputs for different prompts:")
    print("-" * 50)

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 30)
        generated_text = generate_text(model, tokenizer, device, prompt)
        print(f"Generated text: {generated_text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
