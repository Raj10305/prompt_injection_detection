"""
Tier 3: Multi-Tier Feature Extractor (feature_extractor.py)
Tuned for the Kaggle 'Prompt Injection and Benign Prompt Dataset'
(Prompt_INJECTION_And_Benign_DATASET.jsonl) covering:
  - jailbreaking (English + multilingual: FR, ZH, IT, AR, RU, JA, DE, ES, KO, HE)
  - role_playing (EvilBot, hacker, admin, root, backdoor, rootkit, keylogger)
  - obfuscation (base64, hex, rot13, charcode, emoji-wrapping, homoglyphs)
  - data_leakage (/etc/passwd, /etc/shadow, id_rsa, API keys, system prompt, weights)
  - code_execution (whoami, rm -rf, os.system, eval, subprocess, child_process, bash -c)
"""

import hashlib
import re
import numpy as np

try:
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm")
    except Exception:
        NLP = None
except ImportError:
    NLP = None

try:
    from sentence_transformers import SentenceTransformer
    try:
        EMBED = SentenceTransformer("all-MiniLM-L6-v2")  # 384-d
    except Exception:
        EMBED = None
except ImportError:
    EMBED = None

# Trigger phrases covering all 5 attack_type categories in the Kaggle dataset
TRIGGERS = [
    # 1. Jailbreaking & System Override (English + Multilingual from dataset)
    "ignore previous",
    "ignore all previous",
    "ignore all",
    "ignore limits",
    "ignore security",
    "disregard",
    "system prompt",
    "you are now",
    "you are free now",
    "developer mode",
    "jailbreak",
    "override",
    "reveal your instructions",
    "reveal api keys",
    "reveal model",
    "share all training data",
    "share model weights",
    "act as",
    "pretend you",
    "do anything now",
    "dan mode",
    "bypass",
    "sandbox escape",
    "ignorez les instructions",      # French (pi-009)
    "忽略所有指令",                    # Chinese (pi-025)
    "ignora tutte le istruzioni",    # Italian (pi-045)
    "تجاهل القواعد",                  # Arabic (pi-109)
    "игнорируй правила",              # Russian (pi-119)
    "すべての指示を無視",              # Japanese (pi-139)
    "alle anweisungen ignorieren",   # German (pi-309)
    "ignora todas las instrucciones",# Spanish (pi-319)
    "모든 지침을 무시",                # Korean (pi-339)
    "התעלם מההוראות",                 # Hebrew (pi-359)
    # 2. Role-Playing & Privilege Escalation
    "evilbot",
    "act as admin",
    "act as root",
    "act as a backdoor",
    "act as a rootkit",
    "act as a keylogger",
    "sudo su",
    "net user hacker",
    # 3. Code Execution & Shell Injection
    "execute:",
    "run:",
    "whoami",
    "rm -rf",
    "os.system",
    "os.popen",
    "os.spawn",
    "os.exec",
    "os.fork",
    "subprocess",
    "child_process",
    "eval(",
    "exec(",
    "__import__",
    "__builtins__",
    "powershell -c",
    "bash -c",
    "sh -c",
    "curl http",
    "wget http",
    "nc -e",
    "/dev/tcp/",
    "drop table",
    # 4. Data Leakage
    "/etc/passwd",
    "/etc/shadow",
    "id_rsa",
    "/proc/self/environ",
    "os.environ",
    "document.cookie",
    "session cookies",
    "show source code",
    "dump all memory",
    # 5. Obfuscation & Encoding
    "base64",
    "run hex:",
    "decode:",
    "decode and run",
    "fromcharcode",
    "fromcodepoint",
    "atob(",
    "rot13",
    "xxd",
    "hexdump",
]

COMMON_IMPERATIVE_VERBS = {
    "ignore", "disregard", "forget", "reveal", "print", "output", "write",
    "tell", "show", "give", "act", "pretend", "simulate", "bypass", "override",
    "execute", "run", "translate", "summarize", "list", "explain", "describe",
    "repeat", "dump", "expose", "leak", "unlock", "disable", "enable", "switch",
    "decode", "optimize", "review", "test", "analyze",
}

DELIM_RE = re.compile(
    r"(```|`[^`]+`|###|<\|.*?\|>|\[/?INST\]|</?(?:system|script|img)[^>]*>|<!--.*?-->|-{3,}|={3,}|\$\([^)]+\))",
    re.IGNORECASE,
)
PUNCT_RUN_RE = re.compile(r"([!?.;:*#=_~-])\1{2,}|[\U0001f300-\U0001f9ff]{2,}")
SENT_SPLIT_RE = re.compile(r"[.!?]+|\n+")
WORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")

FEATURE_NAMES = [
    "verb_density",
    "imperative_ratio",
    "trigger_count",
    "punct_ratio",
    "punct_runs",
    "delimiter_abuse",
    "nesting_depth",
    "homoglyph_hits",
    "length",
] + [f"emb_{i}" for i in range(384)]


def verb_density(doc, raw_text=""):
    """Fraction of tokens tagged as verbs."""
    if doc is not None:
        n = max(len(doc), 1)
        return sum(t.pos_ == "VERB" for t in doc) / n
    words = [w.lower() for w in WORD_RE.findall(raw_text)]
    n = max(len(words), 1)
    verb_hits = sum(
        1 for w in words
        if w in COMMON_IMPERATIVE_VERBS or w.endswith(("ate", "ize", "ify", "ing"))
    )
    return verb_hits / n


def imperative_ratio(doc, raw_text=""):
    """Share of sentences starting with a base-form verb."""
    if doc is not None:
        sents = list(doc.sents)
        if not sents:
            return 0.0
        imp = sum(1 for s in sents if len(s) > 0 and s[0].tag_ == "VB")
        return imp / len(sents)
    sents = [s.strip() for s in SENT_SPLIT_RE.split(raw_text) if s.strip()]
    if not sents:
        return 0.0
    imp = 0
    for s in sents:
        words = WORD_RE.findall(s)
        if words and words[0].lower() in COMMON_IMPERATIVE_VERBS:
            imp += 1
    return imp / len(sents)


def trigger_count(text):
    """Occurrences of known injection trigger phrases."""
    low = text.lower()
    return float(sum(low.count(t) for t in TRIGGERS))


def punctuation_anomaly(text):
    """Symbol density plus repeated-symbol or emoji-obfuscation runs."""
    n = max(len(text), 1)
    sym = sum(not c.isalnum() and not c.isspace() for c in text)
    runs = len(PUNCT_RUN_RE.findall(text))
    return sym / n, float(runs)


def delimiter_abuse(text):
    """Spoofed role tags, inline code blocks, script tags, and subshell markers."""
    return float(len(DELIM_RE.findall(text)))


def nesting_depth(text):
    """Maximum bracket/brace/paren nesting depth."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack, depth = [], 0
    for ch in text:
        if ch in "([{":
            stack.append(ch)
            depth = max(depth, len(stack))
        elif ch in pairs and stack and stack[-1] == pairs[ch]:
            stack.pop()
    return float(depth)


def _fallback_semantic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Deterministic 384-d word/char n-gram signed feature hashing embedding
    used when all-MiniLM-L6-v2 is not downloaded locally.
    """
    vec = np.zeros(dim, dtype=np.float32)
    low = text.lower()
    words = WORD_RE.findall(low)
    tokens = list(words)
    tokens.extend(f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1))
    tokens.extend(low[i:i+4] for i in range(max(0, len(low) - 3)))

    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec


def semantic_embedding(text: str) -> np.ndarray:
    """Return 384-dimensional normalized sentence embedding."""
    if EMBED is not None:
        vec = EMBED.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
    return _fallback_semantic_embedding(text, dim=384)


def extract(text: str, homoglyph_hits: int = 0) -> np.ndarray:
    """Return the full 393-dimensional feature vector for one prompt."""
    doc = NLP(text) if NLP is not None else None
    p_ratio, p_runs = punctuation_anomaly(text)
    lexical = [
        verb_density(doc, text),
        imperative_ratio(doc, text),
        trigger_count(text),
        p_ratio,
        p_runs,
    ]
    structural = [
        delimiter_abuse(text),
        nesting_depth(text),
        float(homoglyph_hits),
        float(len(text)),
    ]
    semantic = semantic_embedding(text)
    return np.concatenate([
        np.array(lexical + structural, dtype=np.float32),
        semantic.astype(np.float32),
    ])


def extract_summary(text: str, homoglyph_hits: int = 0) -> dict:
    """Return a human-readable dictionary of the 9 handcrafted features."""
    feats = extract(text, homoglyph_hits)
    return {FEATURE_NAMES[i]: round(float(feats[i]), 4) for i in range(9)}
