import torch
from transformers import AutoTokenizer

from model import DeepSeekForCausalLM, ModelConfig


def load_trained_model(checkpoint_path):
    # Initialize model and config
    config = ModelConfig()
    model = DeepSeekForCausalLM(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

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
        generated_tokens = input_ids[0].tolist()

        for _ in range(max_length):
            outputs = model(input_ids)
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
                0, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)

            generated_tokens.append(next_token.item())
            next_token = next_token.unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


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
