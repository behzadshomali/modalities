import torch
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from recursive_llama import RecursiveLlamaConfig, RecursiveLlamaForCausalLM
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="Compare base vs recursive model logits during training")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--recursion_start", type=int, default=0, help="Recursion start layer")
    parser.add_argument("--recursion_end", type=int, default=0, help="Recursion end layer")
    parser.add_argument("--num_recursions", type=int, default=1, help="Number of recursions")
    args = parser.parse_args()

    device = "cuda:1"

    print("Loading base model...")
    base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    print("Creating recursive model...")
    config = RecursiveLlamaConfig(
        model_name="meta-llama/Llama-3.2-1B",
        original_num_hidden_layers=base_model.config.num_hidden_layers,
        recursion_start_layer=args.recursion_start,
        recursion_end_layer=args.recursion_end,
        num_recursions=args.num_recursions,
        sample_random_recursion=False,
        track_diagnostics=False,
    )
    model = RecursiveLlamaForCausalLM(config)

    # Reload clean base model (no recursion)
    del base_model
    base_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    # Move both to device
    base_model, model = base_model.to(device), model.to(device)
    base_model.train()
    model.train()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    base_optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr)

    texts = [
        "From the details given, the car has traveled 5 meters at the 1st turn + 8 meters after the 2nd turn + 0 meters after the 4th turn = <<5+8+0=13>>13 meters around the ring. It must therefore have driven 23 total meters – 13 calculated meters = 10 meters after the 3rd turn. #### 10",
        "The quick brown fox jumps over the lazy dog.",
        "Transformers are powerful models for language processing.",
        "Deep learning enables amazing applications.",
        "GPT models can generate realistic text.",
        "He eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32 He eats 16 from the small pizza because 2 x 8 = <<2*8=16>>16 He eats 48 pieces because 32 + 16 = <<32+16=48>>48 #### 48",

    ]

    for step in range(args.num_steps):
        idx = torch.randint(0, len(texts), (args.batch_size,))
        batch_texts = [texts[i] for i in idx]

        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        labels = inputs["input_ids"].clone()


        base_outputs = base_model(**inputs, labels=labels, use_cache=False)
        base_loss = base_outputs.loss
        base_logits = base_outputs.logits
        base_optimizer.zero_grad()
        base_loss.backward()

        outputs = model(**inputs, labels=labels, use_cache=False)
        loss = outputs.loss
        logits = outputs.logits
        optimizer.zero_grad()
        loss.backward()

        # === Compare ===
        diff = (logits - base_logits).float()
        l2_diff = diff.pow(2).mean().sqrt().item()
        cos_sim = F.cosine_similarity(
            logits.flatten(start_dim=1), base_logits.flatten(start_dim=1), dim=1
        ).mean().item()

        loss_diff = (loss - base_loss).item()

        grad_diff = 0.0
        for p, bp in zip(model.parameters(), base_model.parameters()):
            if p.grad is not None and bp.grad is not None:
                grad_diff += (p.grad - bp.grad).pow(2).sum().item()

        print(f"\n=== Step {step+1}/{args.num_steps} ===")
        print(f"Loss: {loss.item():.4f}")
        print(f"Logits shape: {logits.shape}")
        print(f"L2 diff vs base: {l2_diff:.6f}")
        print(f"Loss diff vs base: {loss_diff:.6f}")
        print(f"Grad diff vs base: {grad_diff:.6f}")
        print(f"Cosine similarity vs base: {cos_sim:.6f}")
        print("First token logits (truncated):", logits[0, 0, :10].detach().cpu().numpy())
        print("First token base logits (truncated):", base_logits[0, 0, :10].detach().cpu().numpy())

        optimizer.step()
        base_optimizer.step()


if __name__ == "__main__":
    main()
