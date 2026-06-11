"""Few-shot domain-recovery experiment: fine-tune a hotel-review detector on a SMALL number
of labeled out-of-domain (RAID) examples and measure how much OOD performance recovers.

Protocol:
  * The OOD parquet is split ONCE (fixed seed 0, stratified by label) into an adaptation
    pool (60%) and a held-out OOD test set (40%); the test set never changes across runs.
  * For each (shots, seed) the checkpoint is reloaded fresh, a stratified sample of size
    ``shots`` is drawn from the pool with that seed (seed controls ONLY the sample), the
    model is fine-tuned for a few epochs, and the held-out OOD test set is evaluated.
  * shots=0 is the un-adapted reference: evaluation only. It is deterministic, so it is
    run once rather than once per seed.

Adaptation uses plain cross-entropy for ALL variants — including softreg, whose attribution
penalty is deliberately NOT applied here. The question is how much a few in-domain labels
recover performance under ordinary fine-tuning; the hard-mask invariance still holds because
masking lives in the collator, which is shared between adaptation and evaluation.

Example:
  python scripts/fewshot_adapt.py --checkpoint results/cells/hardmask_s0_model.pt \
      --variant hardmask --ood_parquet results/cache/raid_ood.parquet --shots 0 50 100 200
"""
import argparse
import gc
import os
from dataclasses import asdict

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from faithdetect.data.maide_up import TextLabelDataset
from faithdetect.evaluate import classification_metrics, predict
from faithdetect.function_words import build_function_word_set
from faithdetect.models import ModelConfig, ReviewDetector, build_tokenizer
from faithdetect.train import make_collator
from faithdetect.utils.logging import env_info, save_json
from faithdetect.utils.seeding import seed_worker, set_seed
from faithdetect.utils.stats import aggregate_seeds

POOL_SPLIT_SEED = 0   # the pool/test split is fixed regardless of --seeds
POOL_FRACTION = 0.6


def split_pool_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One fixed, stratified split into adaptation pool (60%) and held-out test (40%)."""
    pool, test = train_test_split(
        df, train_size=POOL_FRACTION, random_state=POOL_SPLIT_SEED, stratify=df["label"]
    )
    return pool.reset_index(drop=True), test.reset_index(drop=True)


def draw_shots(pool: pd.DataFrame, shots: int, seed: int) -> pd.DataFrame:
    """Stratified sample of size `shots` from the adaptation pool (seed varies the sample)."""
    if shots >= len(pool):
        return pool.reset_index(drop=True)
    sample, _ = train_test_split(
        pool, train_size=shots, random_state=seed, stratify=pool["label"]
    )
    return sample.reset_index(drop=True)


def load_model(checkpoint: str, model_cfg: ModelConfig, vocab_size: int, device) -> ReviewDetector:
    model = ReviewDetector(model_cfg, vocab_size=vocab_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device)


def adapt(
    model: ReviewDetector,
    sample: pd.DataFrame,
    collator,
    device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> None:
    """Plain cross-entropy fine-tuning (no soft-reg penalty, no scheduler), in place."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TextLabelDataset.from_frame(sample),
        batch_size=batch_size, shuffle=True, collate_fn=collator,
        worker_init_fn=seed_worker, generator=g,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model: ReviewDetector, test: pd.DataFrame, collator, device, batch_size: int) -> dict:
    pred = predict(model, test, collator, device, batch_size=batch_size)
    return classification_metrics(pred["y_true"], pred["y_pred"], pred["p_ai"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="trained state_dict .pt")
    ap.add_argument("--variant", required=True, choices=["baseline", "hardmask", "softreg"])
    ap.add_argument("--encoder", default="distilroberta-base")
    ap.add_argument("--ood_parquet", required=True, help="RAID OOD parquet with text,label")
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 50, 100, 200])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    out_json = args.out_json or os.path.join("results", f"fewshot_{args.variant}.json")

    device = torch.device(args.device)
    ood = pd.read_parquet(args.ood_parquet)
    pool, test = split_pool_test(ood)
    print(f"OOD: {len(ood)} rows -> pool {len(pool)} / test {len(test)} "
          f"(fixed split, seed {POOL_SPLIT_SEED})")

    fw_set = build_function_word_set("union")
    tokenizer, func_id = build_tokenizer(args.encoder)
    model_cfg = ModelConfig(
        encoder_name=args.encoder, variant=args.variant, max_length=args.max_length
    )
    # Same collator the variant trained with: hard masking iff variant == 'hardmask'.
    collator = make_collator(model_cfg, tokenizer, func_id, fw_set)

    results: dict[str, dict] = {}
    for shots in args.shots:
        # shots=0 is evaluation-only and deterministic; one run suffices.
        seeds = args.seeds[:1] if shots == 0 else args.seeds
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            model = load_model(args.checkpoint, model_cfg, len(tokenizer), device)
            if shots > 0:
                sample = draw_shots(pool, shots, seed)
                adapt(model, sample, collator, device,
                      epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=seed)
            metrics = evaluate(model, test, collator, device, batch_size=32)
            metrics["seed"] = seed
            per_seed.append(metrics)
            print(f"[shots {shots}|seed {seed}] f1_macro={metrics['f1_macro']:.4f} "
                  f"acc={metrics['accuracy']:.4f}")
            del model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        results[str(shots)] = {"per_seed": per_seed, "aggregated": aggregate_seeds(per_seed)}

    payload = {
        "config": {
            "checkpoint": args.checkpoint, "variant": args.variant, "encoder": args.encoder,
            "ood_parquet": args.ood_parquet, "shots": args.shots, "epochs": args.epochs,
            "lr": args.lr, "batch_size": args.batch_size, "max_length": args.max_length,
            "seeds": args.seeds, "device": args.device,
            "pool_split_seed": POOL_SPLIT_SEED, "pool_fraction": POOL_FRACTION,
            "n_pool": len(pool), "n_test": len(test),
            "model_config": asdict(model_cfg),
        },
        "env": env_info(),
        "results": results,
    }
    save_json(out_json, payload)
    print(f"\nSaved -> {out_json}")

    print("\nshots -> mean macro-F1 on held-out OOD test:")
    for shots in args.shots:
        agg = results[str(shots)]["aggregated"].get("f1_macro")
        if agg:
            print(f"  {shots:>5} : {agg['mean']:.4f} (sd {agg['sd']:.4f}, n={agg['n']})")


if __name__ == "__main__":
    main()
