import os
import time
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model import DeepSeekForCausalLM


def count_parameters(model):
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    """Calculate the model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


class ModelConfig:
    def __init__(self):
        self.vocab_size = 49152
        self.hidden_size = 576
        self.intermediate_size = 1536
        self.num_hidden_layers = 30
        self.num_attention_heads = 9
        self.num_experts = 8
        self.top_k = 2
        self.hidden_act = "gelu"
        self.max_position_embeddings = 512
        self.initializer_range = 0.041666666666666664
        self.rms_norm_eps = 1e-5
        self.tie_word_embeddings = True
        self.pad_token_id = None
        self.bos_token_id = 0
        self.eos_token_id = 0


def prepare_dataset():
    # Load dataset
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return dataset, tokenizer


def train_model(
    model, dataset, tokenizer, num_steps=10100, batch_size=4, learning_rate=1e-4
):
    # Move model to device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Print model statistics
    total_params = count_parameters(model)
    model_size = get_model_size(model)
    print("\nModel Statistics:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print("-" * 50)

    # Initialize optimizer with a slightly higher learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95)
    )

    # Use gradient accumulation to effectively increase batch size
    gradient_accumulation_steps = 4
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Using gradient accumulation: {gradient_accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")

    # Enable mixed precision training if on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    use_amp = scaler is not None
    if use_amp:
        print("Using mixed precision training")

    # Training loop
    model.train()
    step = 0
    total_loss = 0
    last_print_step = 0
    last_time = time.time()
    optimizer.zero_grad()

    print(f"Starting training for {num_steps} steps...")
    print(f"Using device: {device}")

    # Get dataset iterator
    train_iter = iter(dataset["train"])

    while step < num_steps:
        # Get batch
        batch_texts = []
        for _ in range(batch_size):
            try:
                item = next(train_iter)
                batch_texts.append(item["text"])
            except StopIteration:
                train_iter = iter(dataset["train"])
                item = next(train_iter)
                batch_texts.append(item["text"])

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Add labels for causal language modeling
        inputs["labels"] = inputs["input_ids"].clone()

        # Forward pass with mixed precision if available
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
                loss = outputs / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        else:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = outputs / gradient_accumulation_steps
            loss.backward()

        # Track loss
        total_loss += loss.item() * gradient_accumulation_steps

        # Update weights after accumulating gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

        # Update progress
        step += 1

        # Print detailed stats every 10 steps
        if step - last_print_step >= 10:
            current_time = time.time()
            time_taken = current_time - last_time
            avg_loss = total_loss / step
            print(
                f"\nStep {step}/{num_steps} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | Average Loss: {avg_loss:.4f} | Time for 10 steps: {time_taken:.2f}s"
            )
            last_print_step = step
            last_time = current_time

        # Save checkpoint every 500 steps
        if step % 500 == 0:
            checkpoint_path = f"checkpoint_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item() * gradient_accumulation_steps,
                },
                checkpoint_path,
            )
            print(f"\nSaved checkpoint to {checkpoint_path}")

    print("\nTraining completed!")
    return model


def main():
    # Initialize model and config
    config = ModelConfig()
    model = DeepSeekForCausalLM(config)

    # Prepare dataset and tokenizer
    dataset, tokenizer = prepare_dataset()

    # Train model
    model = train_model(model, dataset, tokenizer)

    # Save final model
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config}, "final_model.pt"
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
