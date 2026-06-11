"""Generate same-domain hotel reviews with LOCAL open-weight LLMs served by Ollama.

Why: MAiDE-up's AI class is GPT-4 only, and the RAID pool covers other generators only in
OTHER domains. To test whether the trained detectors generalize to NEW open-weight
generators in the SAME domain (hotel reviews), we prompt local models through Ollama with
real MAiDE-up hotel names and cities, producing reviews that differ from the training AI
class only in the generator.

Protocol
--------
* Ensures an Ollama server is reachable at ``--host`` (spawns ``ollama serve`` in the
  background and polls if not), and pulls any model missing from ``/api/tags``.
* Rotates over ~20 prompt templates instantiated with real (hotel, city) pairs drawn from
  ``data/all_data.csv`` (columns 'Hotel Name', 'City Name'), varying sentiment, traveler
  context, and the concrete details requested (3-6 sentences each).
* Sampling uses temperature 0.9 with a distinct per-sample seed passed to ``/api/generate``
  (stream=false), so outputs are varied yet reproducible for a fixed (model, prompt, seed).
* Outputs shorter than 20 words are discarded; leading meta-text such as
  "Here is a review:" is stripped; exact duplicates are dropped.
* Rows {text, label=1, model} are checkpointed to parquet every 20 kept samples and the
  script RESUMES from an existing parquet, so it is cache-friendly and safe to re-run
  niced or after an interruption.

Example:
  python scripts/generate_ollama.py --models llama3.2:3b gemma2:2b --n_per_model 100 \
      --out results/cache/ollama_reviews.parquet
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import time

import pandas as pd
import requests

import _bootstrap  # noqa: F401

AI_LABEL = 1
MIN_WORDS = 20
CHECKPOINT_EVERY = 20

# Every template asks for a short first-person guest review with concrete details; the
# scenario (sentiment, traveler type, what to mention) varies across templates so the
# generated class is not a single stylistic mode.
SUFFIX = (" Write 3-6 sentences in the first person. Output only the review text itself - "
          "no title, no preamble, no quotation marks.")
TEMPLATES: tuple[str, ...] = tuple(t + SUFFIX for t in (
    'Write a guest review of the hotel "{hotel}" in {city}. You had a wonderful stay; '
    'mention the comfortable room and the friendly front-desk staff.',
    'Write a guest review of "{hotel}" in {city}. You were disappointed: the room was '
    'noisy and housekeeping was slow. Mention specific problems.',
    'Write a guest review of "{hotel}" in {city} after a business trip. Comment on the '
    'wifi, the desk in the room, and the breakfast.',
    'Write a guest review of "{hotel}" in {city} from a family vacation with two kids. '
    'Mention the pool and how the staff treated the children.',
    'Write a mixed guest review of "{hotel}" in {city}: great location but dated rooms. '
    'Mention one thing you would change.',
    'Write a guest review of "{hotel}" in {city}. The check-in took too long but the rest '
    'of the stay was pleasant. Mention the bed and the view.',
    'Write a glowing guest review of "{hotel}" in {city} for an anniversary weekend. '
    'Mention a specific touch the staff added.',
    'Write a guest review of "{hotel}" in {city} as a solo traveler. Mention safety, the '
    'neighborhood, and the breakfast buffet.',
    'Write a critical guest review of "{hotel}" in {city}: the photos online looked '
    'better than reality. Mention two concrete letdowns.',
    'Write a guest review of "{hotel}" in {city} praising value for money. Mention the '
    'price, the room size, and the location.',
    'Write a guest review of "{hotel}" in {city}. You stayed one night before an early '
    'flight; comment on the airport shuttle and how quiet the room was.',
    'Write a guest review of "{hotel}" in {city} from a returning guest who has stayed '
    'three times. Mention what keeps you coming back.',
    'Write a lukewarm three-star guest review of "{hotel}" in {city}. Nothing terrible, '
    'nothing special; mention the average breakfast.',
    'Write a guest review of "{hotel}" in {city} complaining about the air conditioning '
    'and thin walls, but praising the restaurant downstairs.',
    'Write a guest review of "{hotel}" in {city} after attending a conference there. '
    'Mention the meeting facilities and the coffee.',
    'Write an enthusiastic guest review of "{hotel}" in {city} focused on the rooftop bar '
    'and the view at sunset. Mention a staff member who helped you.',
    'Write a guest review of "{hotel}" in {city} from a couple on a weekend city break. '
    'Mention walkability, the bed, and the late checkout.',
    'Write a guest review of "{hotel}" in {city}. Your booking was mixed up at arrival; '
    'describe how the staff resolved it and give your overall verdict.',
    'Write a guest review of "{hotel}" in {city} focused on the spa and the gym. Mention '
    'opening hours and cleanliness.',
    'Write a short, slightly grumpy guest review of "{hotel}" in {city}: good location, '
    'overpriced minibar, slow elevator. End on whether you would return.',
))

# Leading meta-text the instruct models sometimes prepend despite the instruction.
META_PATTERNS: tuple[str, ...] = (
    r"^(?:sure|okay|certainly|of course|absolutely)[!,.:]?\s+",
    r"^here(?:'s| is)\b[^:\n]{0,90}:\s*",
    r"^(?:guest |hotel |my |short )?review(?: text)?:\s*",
)


def server_alive(host: str, timeout: float = 3.0) -> bool:
    """True iff an Ollama server answers /api/tags at `host`."""
    try:
        return requests.get(f"{host}/api/tags", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def ensure_server(host: str, wait_s: float = 60.0) -> None:
    """Make sure an Ollama server is reachable; spawn ``ollama serve`` and poll if not.

    A spawned server is intentionally left running (other jobs on this machine may use it);
    its output goes to results/cache/ollama_serve.log.
    """
    if server_alive(host):
        print(f"Ollama server already up at {host}")
        return
    os.makedirs("results/cache", exist_ok=True)
    log = open("results/cache/ollama_serve.log", "ab")
    print(f"No server at {host}; spawning 'ollama serve' (log: results/cache/ollama_serve.log)")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, start_new_session=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if server_alive(host):
            print("Ollama server is up.")
            return
        time.sleep(1.0)
    raise RuntimeError(f"Ollama server did not come up at {host} within {wait_s:.0f}s")


def installed_models(host: str) -> set[str]:
    """Names of locally available models per /api/tags (e.g. 'llama3.2:3b')."""
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    return {m.get("name", "") for m in r.json().get("models", [])}


def ensure_model(host: str, model: str) -> None:
    """Pull `model` via the ollama CLI if it is not in /api/tags (exact or base-name match)."""
    names = installed_models(host)
    if model in names or (":" not in model and any(n.split(":")[0] == model for n in names)):
        return
    print(f"[pull] '{model}' not installed; running 'ollama pull {model}' ...")
    subprocess.run(["ollama", "pull", model], check=True)


def ollama_generate(host: str, model: str, prompt: str, temperature: float, seed: int,
                    timeout: float = 300.0) -> str:
    """One non-streaming /api/generate call; returns the raw completion text."""
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": 256},
    }
    r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return str(r.json().get("response", ""))


def clean_review(raw: str) -> str:
    """Strip leading meta-text/titles and wrapping quotes; collapse whitespace.

    Conservative by design: only clearly non-review prefixes (lines ending in ':',
    "Here is ...:" lead-ins, markdown title lines) are removed, never review content.
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    while lines:
        head = lines[0].strip("*#_ ").rstrip()
        if (head.endswith(":") and len(head) <= 90) or re.fullmatch(r"\*{1,2}[^*]{1,80}\*{1,2}", lines[0]):
            lines.pop(0)
            continue
        break
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    for pat in META_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text


def load_hotel_city_pairs(csv_path: str) -> list[tuple[str, str]]:
    """Unique, deterministically ordered (hotel, city) pairs from the MAiDE-up csv."""
    df = pd.read_csv(csv_path, usecols=["Hotel Name", "City Name"]).dropna()
    pairs = sorted({(str(h).strip(), str(c).strip())
                    for h, c in zip(df["Hotel Name"], df["City Name"])})
    if not pairs:
        raise ValueError(f"No (hotel, city) pairs found in {csv_path}")
    return pairs


def write_checkpoint(rows: list[dict], out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=["text", "label", "model"]).to_parquet(out, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["llama3.2:3b", "gemma2:2b"],
                    help="Ollama model tags to generate with")
    ap.add_argument("--n_per_model", type=int, default=100)
    ap.add_argument("--out", default="results/cache/ollama_reviews.parquet")
    ap.add_argument("--data_csv", default="data/all_data.csv",
                    help="MAiDE-up csv supplying real hotel names and cities")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0, help="base seed; per-sample seeds derive from it")
    ap.add_argument("--request_timeout", type=float, default=300.0)
    args = ap.parse_args()

    ensure_server(args.host)
    pairs = load_hotel_city_pairs(args.data_csv)
    print(f"{len(pairs)} (hotel, city) pairs, {len(TEMPLATES)} prompt templates")

    rows: list[dict] = []
    if os.path.exists(args.out):
        existing = pd.read_parquet(args.out)
        rows = existing.to_dict("records")
        print(f"Resuming: {len(rows)} rows already in {args.out}")
    seen_texts = {r["text"] for r in rows}
    done = pd.Series([r["model"] for r in rows]).value_counts().to_dict() if rows else {}

    for mi, model in enumerate(args.models):
        kept = int(done.get(model, 0))
        if kept >= args.n_per_model:
            print(f"[{model}] already complete ({kept}/{args.n_per_model}); skipping")
            continue
        ensure_model(args.host, model)
        attempt, skipped, fail_streak = kept, 0, 0
        max_attempts = 5 * args.n_per_model
        print(f"[{model}] generating {args.n_per_model - kept} reviews "
              f"(temperature={args.temperature})")
        while kept < args.n_per_model and attempt < max_attempts:
            sample_seed = args.seed * 1_000_000 + mi * 100_000 + attempt
            rng = random.Random(sample_seed)
            hotel, city = rng.choice(pairs)
            prompt = TEMPLATES[attempt % len(TEMPLATES)].format(hotel=hotel, city=city)
            attempt += 1
            try:
                raw = ollama_generate(args.host, model, prompt, args.temperature,
                                      sample_seed, timeout=args.request_timeout)
                fail_streak = 0
            except requests.RequestException as e:
                fail_streak += 1
                print(f"  [warn] request failed ({type(e).__name__}); "
                      f"streak {fail_streak}/5, retrying in 2s")
                if fail_streak >= 5:
                    raise RuntimeError(f"5 consecutive request failures for {model}") from e
                time.sleep(2.0)
                continue
            text = clean_review(raw)
            if len(text.split()) < MIN_WORDS or text in seen_texts:
                skipped += 1
                continue
            rows.append({"text": text, "label": AI_LABEL, "model": model})
            seen_texts.add(text)
            kept += 1
            if kept % 10 == 0 or kept == args.n_per_model:
                print(f"  [{model}] kept {kept}/{args.n_per_model} (skipped {skipped})")
            if len(rows) % CHECKPOINT_EVERY == 0:
                write_checkpoint(rows, args.out)
        if kept < args.n_per_model:
            print(f"  [warn] [{model}] only {kept}/{args.n_per_model} kept "
                  f"after {max_attempts} attempts")
        write_checkpoint(rows, args.out)

    write_checkpoint(rows, args.out)
    df = pd.DataFrame(rows)
    print(f"\nSaved {len(df)} rows -> {args.out}")
    if len(df):
        print(df["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
