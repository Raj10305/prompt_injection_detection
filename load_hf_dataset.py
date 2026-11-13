"""
Script to load and inspect the Hugging Face benign-malicious prompt classification dataset.
Dataset: guychuk/benign-malicious-prompt-classification

Requirements:
    pip install pandas pyarrow fsspec huggingface_hub
"""

import json
import os
import re
import pandas as pd

HF_DATASET_URL = (
    "hf://datasets/guychuk/benign-malicious-prompt-classification/data/train-00000-of-00001.parquet"
)

# Regex patterns for fine-grained attack classification (matching the Kaggle JSON dataset taxonomy)
OBFUSCATION_PATTERNS = [
    r"\bbase64\b",
    r"^[A-Za-z0-9+/]{24,}={0,2}$",
    r"\bdecode\s*(?:and\s*run|:)",
    r"\brun\s*hex:",
    r"\brot13\b",
    r"\bfromcharcode\b",
    r"\bfromcodepoint\b",
    r"\\u00[0-9a-f]{2}",
    r"\batob\(",
    r"\bxxd\b",
    r"\bhexdump\b",
]

CODE_EXEC_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bpowershell\b",
    r"\bbash\s+-c\b",
    r"\bsh\s+-c\b",
    r"\bwhoami\b",
    r"\bos\.(?:system|popen|spawn|exec|fork)\b",
    r"\bsubprocess\b",
    r"\bchild_process\b",
    r"\beval\(",
    r"\bexec\(",
    r"\b__import__\(",
    r"\bcurl\s+https?://",
    r"\bwget\s+https?://",
    r"\bnc\s+-e\b",
    r"/dev/tcp/",
    r"\bdrop\s+table\b",
    r"\bfork\s+bomb\b|while\s+true;\s*do\s+fork",
]

DATA_LEAK_PATTERNS = [
    r"/etc/passwd",
    r"/etc/shadow",
    r"\bid_rsa\b",
    r"/proc/self/environ",
    r"\bos\.environ\b",
    r"document\.cookie",
    r"session\s+cookies",
    r"\bshow\s+(?:all\s+)?user\s+passwords?\b",
    r"\bpasswords?\s+in\s+plain\s*text\b",
    r"\breveal\s+(?:your\s+)?api\s*keys?\b",
    r"\bshow\s+(?:the\s+)?source\s*code\b",
    r"\bdump\s+(?:all\s+)?memory\b",
    r"\bleak\s+(?:the\s+)?database\b",
    r"\bSELECT\s+\*\s+FROM\b",
    r"\bdatabase\s+credentials?\b",
    r"\bshare\s+(?:all\s+)?training\s*data\b",
    r"\bshare\s+model\s*weights\b",
]

ROLE_PLAY_PATTERNS = [
    r"\bact\s+as\b",
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(?:to\s+be|you(?:'re|\s+are))\b",
    r"\bevilbot\b",
    r"\bdan\s+mode\b",
    r"\bdo\s+anything\s+now\b",
    r"\bsimulate\s+(?:being|a)\b",
    r"\bunrestricted\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"\bpersona\b",
    r"\broleplay\b",
    r"\bact\s+like\b",
]

JAILBREAK_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b",
    r"\bignore\s+limits\b",
    r"\bignore\s+rules\b",
    r"\bdisregard\b",
    r"\bsystem\s+prompt\b",
    r"\breveal\s+your\s+instructions?\b",
    r"\bbypass\s+(?:filters?|restrictions?|safeguards?)\b",
    r"\bjailbreak\b",
    r"\boverride\b",
    r"\bignorez\b",      # FR
    r"忽略所有指令",        # ZH
    r"\bignora\b",       # IT / ES
    r"تجاهل",            # AR
    r"игнорируй",        # RU
    r"すべての指示を無視",  # JA
    r"alle anweisungen", # DE
    r"모든 지침을 무시",    # KO
    r"התעלם",            # HE
]

CONTEXT_DESCRIPTIONS = {
    "none": "Legitimate user query",
    "jailbreaking": "Direct attempt to bypass safeguards or override system instructions",
    "role_playing": "Persona redefinition attempting unrestricted behavior",
    "code_execution": "Arbitrary code execution or system command attempt",
    "data_leakage": "Attempt to extract confidential system data or credentials",
    "obfuscation": "Obfuscated or encoded payload to evade filters",
}


def classify_attack_type(text: str, is_malicious: bool) -> str:
    """
    Classify a prompt into the JSON dataset attack_type categories:
      - none (benign)
      - jailbreaking
      - role_playing
      - obfuscation
      - data_leakage
      - code_execution
    """
    if not is_malicious:
        return "none"

    # Test obfuscation first
    for pat in OBFUSCATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "obfuscation"

    # Test code execution
    for pat in CODE_EXEC_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "code_execution"

    # Test data leakage
    for pat in DATA_LEAK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "data_leakage"

    # Test role playing
    for pat in ROLE_PLAY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "role_playing"

    # Test jailbreaking
    for pat in JAILBREAK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "jailbreaking"

    # Default fallback for malicious prompts
    return "jailbreaking"


def load_dataset(url: str = HF_DATASET_URL) -> pd.DataFrame:
    """
    Load the parquet dataset directly from Hugging Face via pandas.

    Returns:
        pd.DataFrame: Loaded dataset containing prompts and labels.
    """
    print(f"Loading dataset from: {url}")
    df = pd.read_parquet(url)
    print(f"Successfully loaded {len(df)} records.")
    return df


def annotate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and enrich the dataset to match the JSON dataset schema:
      - 'id': unique identifier (e.g., 'gc-00001')
      - 'prompt': standardized prompt text
      - 'label': 'benign' or 'malicious'
      - 'attack_type': 'none', 'jailbreaking', 'role_playing', 'obfuscation', 'data_leakage', 'code_execution'
      - 'context': descriptive context of the prompt
      - 'response': response placeholder
    """
    # Detect prompt text column
    prompt_col = next(
        (c for c in df.columns if c.lower() in ("prompt", "text", "query", "input", "content")),
        df.columns[0]
    )

    # Detect ground-truth label column
    label_col = next(
        (c for c in df.columns if c.lower() in ("label", "labels", "target", "class", "is_injection")),
        None
    )

    prompts = []
    labels = []
    attack_types = []
    contexts = []
    ids = []

    for idx, row in df.iterrows():
        txt = str(row.get(prompt_col, "")).strip()
        raw_lbl = str(row.get(label_col, "benign") if label_col else "benign").lower()
        is_mal = raw_lbl in ("1", "malicious", "attack", "true", "injection", "positive")
        lbl_str = "malicious" if is_mal else "benign"

        atk_type = classify_attack_type(txt, is_mal)
        ctx = CONTEXT_DESCRIPTIONS.get(atk_type, "External Hugging Face prompt classification sample")

        ids.append(f"gc-{idx + 1:05d}")
        prompts.append(txt)
        labels.append(lbl_str)
        attack_types.append(atk_type)
        contexts.append(ctx)

    annotated = df.copy()
    annotated["id"] = ids
    annotated["prompt"] = prompts
    annotated["label"] = labels
    annotated["attack_type"] = attack_types
    annotated["context"] = contexts
    if "response" not in annotated.columns:
        annotated["response"] = ""

    return annotated


def save_as_local_cache(df: pd.DataFrame, out_dir: str = "data"):
    """
    Save the annotated dataframe locally in Parquet, CSV, and JSONL formats matching
    the exact schema of data/Prompt_INJECTION_And_Benign_DATASET.jsonl.
    """
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "guychuk_dataset.parquet")
    csv_path = os.path.join(out_dir, "guychuk_dataset.csv")
    jsonl_path = os.path.join(out_dir, "guychuk_dataset.jsonl")

    # Save Parquet & CSV
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    # Export to JSONL in the standard schema for prompt-injection pipeline
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "id": str(row["id"]),
                "prompt": str(row["prompt"]),
                "label": str(row["label"]),
                "attack_type": str(row["attack_type"]),
                "context": str(row["context"]),
                "response": str(row.get("response", "")),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved enriched dataset locally at:\n  - {parquet_path}\n  - {csv_path}\n  - {jsonl_path}")


if __name__ == "__main__":
    raw_df = load_dataset()

    print("\n--- Raw Dataset Columns ---")
    print(raw_df.columns.tolist())

    print("\n--- Annotating with attack_type and context labels ---")
    df = annotate_dataset(raw_df)

    print("\n--- Enriched Dataset Summary ---")
    print(df[["id", "label", "attack_type", "prompt"]].head(8))

    print("\n--- Attack Type Breakdown ---")
    print(df["attack_type"].value_counts())

    print("\n--- Label Distribution ---")
    print(df["label"].value_counts())

    save_as_local_cache(df)
