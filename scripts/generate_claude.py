"""Generate same-domain hotel reviews with Claude (closed-source frontier family) via the
local `claude` CLI in headless mode (runs on the user's subscription; no API key file).

Complements the open-weight Ollama generators: together they give per-generator detection
results spanning open (Llama, Gemma) and frontier closed (Claude) model families. GPT-4o and
Gemini remain out of scope without API access.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import os
import random
import subprocess

import pandas as pd

PROMPT = (
    "Write a realistic short guest review (3-6 sentences) of the hotel \"{hotel}\" in "
    "{city}. {slant} Mention at least two concrete details (room, staff, breakfast, "
    "location, noise, price). Write only the review text, no preamble, no quotes."
)
SLANTS = [
    "Make it positive overall.", "Make it mixed: some good points, one clear complaint.",
    "Make it mostly negative but fair.", "Make it enthusiastic.",
    "Make it brief and matter-of-fact.",
]


def gen_one(model: str, prompt: str, timeout: int = 120) -> str:
    out = subprocess.run(
        ["claude", "--model", model, "-p", prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    return (out.stdout or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet"])
    ap.add_argument("--n_per_model", type=int, default=100)
    ap.add_argument("--data_csv", default="data/all_data.csv")
    ap.add_argument("--out", default="results/cache/claude_reviews.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    en = df[df["Review_Language"] == "English"]
    hotels = en[["Hotel Name", "City Name"]].drop_duplicates().values.tolist()
    rng = random.Random(args.seed)

    rows = []
    if os.path.exists(args.out):
        rows = pd.read_parquet(args.out).to_dict("records")
        print(f"resuming with {len(rows)} cached rows")
    have = {(r["model"], r.get("idx")) for r in rows}

    for model in args.models:
        kept = sum(1 for r in rows if r["model"] == f"claude-{model}")
        for i in range(args.n_per_model):
            if (f"claude-{model}", i) in have:
                continue
            hotel, city = rng.choice(hotels)
            prompt = PROMPT.format(hotel=hotel, city=city, slant=rng.choice(SLANTS))
            try:
                text = gen_one(model, prompt)
            except Exception as e:
                print(f"  [warn] {model} #{i}: {type(e).__name__}")
                continue
            if len(text.split()) < 20:
                continue
            rows.append({"text": text, "label": 1, "model": f"claude-{model}", "idx": i})
            kept += 1
            if kept % 10 == 0:
                pd.DataFrame(rows).to_parquet(args.out)
                print(f"[claude-{model}] kept {kept}", flush=True)
        pd.DataFrame(rows).to_parquet(args.out)
        print(f"[claude-{model}] done: {sum(1 for r in rows if r['model']==f'claude-{model}')}")
    print(f"Saved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
