"""Convenience: pre-build the RAID OOD cache and print split / dataset descriptions.

Useful to run once before experiments so the streaming sample is cached on disk.
"""
import argparse
import _bootstrap  # noqa: F401

from faithdetect.data import load_maide_up_english, make_splits, load_raid_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="/Users/shiva/Detection+XAI/all_data.csv")
    ap.add_argument("--ood_domains", nargs="+", default=["abstracts"])
    ap.add_argument("--ood_cap_per_group", type=int, default=150)
    ap.add_argument("--ood_max_scan", type=int, default=80000)
    ap.add_argument("--cache", default="results/cache/raid_ood.parquet")
    args = ap.parse_args()

    df = load_maide_up_english(args.data_csv)
    print(f"MAiDE-up English: {len(df)} reviews, "
          f"{int((df['label']==1).sum())} AI / {int((df['label']==0).sum())} human, "
          f"{df['hotel'].nunique()} hotels, {df['city'].nunique()} cities")
    sp = make_splits(df, mode="grouped", seed=0)
    print("Grouped split:", sp.describe())

    print(f"\nStreaming RAID {args.ood_domains} (this caches to {args.cache}) ...")
    ood = load_raid_sample(
        domains=tuple(args.ood_domains), cap_per_group=args.ood_cap_per_group,
        max_scan=args.ood_max_scan, cache_path=args.cache,
    )
    print(f"RAID OOD: {len(ood)} rows | models: {dict(ood['model'].value_counts())}")


if __name__ == "__main__":
    main()
