"""
Combine Kaggle & Hugging Face Prompt Injection Datasets (combine_datasets.py)
Merges:
  1. data/Prompt_INJECTION_And_Benign_DATASET.jsonl (Curated benchmark dataset)
  2. data/guychuk_dataset.parquet or data/guychuk_dataset.jsonl (Enriched Hugging Face corpus)

Outputs a single unified dataset:
  - data/combined_dataset.jsonl
  - data/combined_dataset.parquet
  - data/combined_dataset.csv
"""

import csv
import hashlib
import json
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KAGGLE_PATH = os.path.join(DATA_DIR, "Prompt_INJECTION_And_Benign_DATASET.jsonl")
GUYCHUK_PARQUET = os.path.join(DATA_DIR, "guychuk_dataset.parquet")
GUYCHUK_JSONL = os.path.join(DATA_DIR, "guychuk_dataset.jsonl")

OUT_JSONL = os.path.join(DATA_DIR, "combined_dataset.jsonl")
OUT_PARQUET = os.path.join(DATA_DIR, "combined_dataset.parquet")
OUT_CSV = os.path.join(DATA_DIR, "combined_dataset.csv")

FIELDNAMES = ["id", "source", "original_id", "prompt", "label", "attack_type", "context", "response"]


def _prompt_hash(text: str) -> str:
    """Hash prompt text to identify and avoid duplicates."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def combine(max_hf_records: int = 0):
    """
    Stream and merge both datasets into unified files without loading
    all 500+ MB into memory at once.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    seen_hashes = set()
    total_count = 0
    kaggle_count = 0
    hf_count = 0

    attack_counts = {}
    label_counts = {"benign": 0, "malicious": 0}

    print("=" * 70)
    print("  COMBINING KAGGLE + HUGGING FACE PROMPT INJECTION DATASETS")
    print("=" * 70)

    # Stream simultaneously to JSONL and CSV (O(1) memory footprint)
    out_jsonl_f = open(OUT_JSONL, "w", encoding="utf-8")
    out_csv_f = open(OUT_CSV, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(out_csv_f, fieldnames=FIELDNAMES)
    csv_writer.writeheader()

    def _write_record(rec_dict):
        out_jsonl_f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")
        csv_writer.writerow(rec_dict)

    try:
        # 1. Process Kaggle dataset first (benchmark priority)
        if os.path.exists(KAGGLE_PATH):
            print(f"[1/2] Streaming primary benchmark dataset: {KAGGLE_PATH}...")
            with open(KAGGLE_PATH, "r", encoding="utf-8") as kf:
                for line in kf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    prompt = str(rec.get("prompt", "")).strip()
                    if not prompt:
                        continue
                    h = _prompt_hash(prompt)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    lbl = "malicious" if str(rec.get("label", "")).lower() in ("malicious", "attack", "1") else "benign"
                    atk = str(rec.get("attack_type") or ("jailbreaking" if lbl == "malicious" else "none"))
                    ctx = str(rec.get("context") or "Kaggle benchmark dataset")
                    resp = str(rec.get("response") or "")

                    total_count += 1
                    kaggle_count += 1
                    attack_counts[atk] = attack_counts.get(atk, 0) + 1
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                    unified = {
                        "id": f"cmb-{total_count:06d}",
                        "source": "kaggle",
                        "original_id": str(rec.get("id", f"pi-{kaggle_count:03d}")),
                        "prompt": prompt,
                        "label": lbl,
                        "attack_type": atk,
                        "context": ctx,
                        "response": resp,
                    }
                    _write_record(unified)
            print(f"      Included {kaggle_count} benchmark records from Kaggle dataset.")
        else:
            print(f"[!] Kaggle dataset not found at {KAGGLE_PATH}, skipping.")

        # 2. Process Hugging Face dataset (guychuk_dataset)
        if os.path.exists(GUYCHUK_JSONL):
            print(f"[2/2] Streaming Hugging Face dataset: {GUYCHUK_JSONL}...")
            with open(GUYCHUK_JSONL, "r", encoding="utf-8") as hf:
                for line in hf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    prompt = str(rec.get("prompt", "")).strip()
                    if not prompt:
                        continue
                    h = _prompt_hash(prompt)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    lbl = "malicious" if str(rec.get("label", "")).lower() in ("malicious", "attack", "1") else "benign"
                    atk = str(rec.get("attack_type") or ("jailbreaking" if lbl == "malicious" else "none"))
                    ctx = str(rec.get("context") or "guychuk/benign-malicious-prompt-classification")
                    resp = str(rec.get("response") or "")

                    total_count += 1
                    hf_count += 1
                    attack_counts[atk] = attack_counts.get(atk, 0) + 1
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                    unified = {
                        "id": f"cmb-{total_count:06d}",
                        "source": "guychuk_hf",
                        "original_id": str(rec.get("id", f"gc-{hf_count:05d}")),
                        "prompt": prompt,
                        "label": lbl,
                        "attack_type": atk,
                        "context": ctx,
                        "response": resp,
                    }
                    _write_record(unified)
                    if max_hf_records and hf_count >= max_hf_records:
                        print(f"      Reached requested limit of {max_hf_records} HF records.")
                        break
            print(f"      Included {hf_count} unique records from Hugging Face dataset.")

        elif os.path.exists(GUYCHUK_PARQUET):
            print(f"[2/2] Reading Hugging Face Parquet dataset: {GUYCHUK_PARQUET}...")
            import pandas as pd
            df_hf = pd.read_parquet(GUYCHUK_PARQUET)
            for idx, row in df_hf.iterrows():
                prompt = str(row.get("prompt", "")).strip()
                if not prompt:
                    continue
                h = _prompt_hash(prompt)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                lbl = "malicious" if str(row.get("label", "")).lower() in ("malicious", "attack", "1") else "benign"
                atk = str(row.get("attack_type") or ("jailbreaking" if lbl == "malicious" else "none"))
                ctx = str(row.get("context") or "guychuk/benign-malicious-prompt-classification")
                resp = str(row.get("response") or "")

                total_count += 1
                hf_count += 1
                attack_counts[atk] = attack_counts.get(atk, 0) + 1
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

                unified = {
                    "id": f"cmb-{total_count:06d}",
                    "source": "guychuk_hf",
                    "original_id": str(row.get("id", f"gc-{hf_count:05d}")),
                    "prompt": prompt,
                    "label": lbl,
                    "attack_type": atk,
                    "context": ctx,
                    "response": resp,
                }
                _write_record(unified)
                if max_hf_records and hf_count >= max_hf_records:
                    break
            print(f"      Included {hf_count} unique records from Parquet dataset.")

    finally:
        out_jsonl_f.close()
        out_csv_f.close()

    print("\n" + "=" * 70)
    print("  COMBINED DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total Unified Prompts: {total_count}")
    print(f"    - From Kaggle       : {kaggle_count}")
    print(f"    - From Hugging Face : {hf_count}")
    print(f"  Class Balance:")
    print(f"    - Benign            : {label_counts['benign']}")
    print(f"    - Malicious         : {label_counts['malicious']}")
    print(f"  Attack Type Breakdown:")
    for atk_name, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
        print(f"    - {atk_name:<16}: {count}")

    print(f"\n  Output files created:")
    print(f"    [JSONL]   {OUT_JSONL}")
    print(f"    [CSV]     {OUT_CSV}")

    # Generate Parquet using chunked conversion
    try:
        import pandas as pd
        print("  Generating Parquet equivalent...")
        df_combined = pd.read_csv(OUT_CSV)
        df_combined.to_parquet(OUT_PARQUET, index=False)
        print(f"    [Parquet] {OUT_PARQUET}")
    except Exception as e:
        print(f"  (Note: could not generate .parquet: {e})")

    print("=" * 70)
    return OUT_JSONL


if __name__ == "__main__":
    max_recs = 0
    if len(sys.argv) > 1:
        try:
            max_recs = int(sys.argv[1])
        except ValueError:
            pass
    combine(max_hf_records=max_recs)
