# FORM 2
THE PATENTS ACT, 1970
(39 of 1970)
&
THE PATENTS RULES, 2003
COMPLETE SPECIFICATION
(See section 10 and rule 13)

---

## 1. TITLE OF THE INVENTION
**A MULTI-TIERED HYBRID SYSTEM AND METHOD FOR REAL-TIME PROMPT INJECTION DETECTION IN LARGE LANGUAGE MODELS**
*(14 words)*

---

## 2. APPLICANT(S)
- **Name**: [Applicant Name / Organization Name]
- **Nationality**: Indian
- **Address**: [Applicant Registered Address]

---

## 3. PREAMBLE TO THE DESCRIPTION

### COMPLETE SPECIFICATION
The following specification particularly describes the invention and the manner in which it is to be performed:

---

## 4. FIELD OF THE INVENTION
The present invention relates generally to the technical domain of artificial intelligence safety, natural language processing security, and cyber-defense architectures. More specifically, the present invention relates to an inline defensive guardrail middleware system `(100)` and computer-implemented method configured to detect, intercept, classify, and sanitize direct and indirect adversarial prompt injection attacks, jailbreak payloads, and systemic instruction overrides targeting Large Language Models (LLMs) prior to inference execution.

---

## 5. BACKGROUND OF THE INVENTION & TECHNICAL PROBLEM
Generative artificial intelligence architectures and Large Language Models (LLMs) operate by evaluating unstructured natural language prompts received from untrusted external users, frequently concatenated with developer-defined system instructions, retrieval-augmented generation (RAG) context, or user-uploaded document payloads. This structural concatenation creates an intrinsic architectural vulnerability known as Prompt Injection, wherein adversarial instructions embedded within the user input subvert the model's instruction hierarchy, causing the LLM to bypass safety alignments, disclose confidential system prompts, leak API credentials, adopt unauthorized personas, or execute arbitrary system commands.

Existing defensive mechanisms manifest significant technological bottlenecks and security vulnerabilities:
1. **Prohibitive Latency of LLM-as-a-Judge Implementations**: Utilizing secondary Large Language Models to inspect primary inputs introduces substantial network and inference latency (typically 800 ms to 2,500 ms per query), rendering such guardrails impractical for real-time interactive applications.
2. **Vulnerability to Character-Level and Multilingual Obfuscation**: Conventional keyword-filtering, heuristic blacklists, and standard regular expressions fail to intercept adversarial perturbations involving Unicode zero-width spaces (`U+200B`–`U+200F`, `U+FEFF`), soft hyphens (`U+00AD`), invisible directional formatting characters (`U+202A`–`U+202E`, `U+2060`–`U+206F`), non-printable control characters, full-width ASCII variants (`U+FF01`–`U+FF5E`), and cross-alphabet Cyrillic/Greek-to-Latin homoglyphic substitutions (e.g., substituting Latin `'a'` with Cyrillic `'а'` `U+0430`).
3. **High False Positive Rates in Standalone Lexical Classifiers**: Purely statistical or keyword-based classifiers frequently misclassify benign technical queries containing imperative verbs or programming terminology (such as *"How do I override the toString() method in Java?"* or *"How to ignore NaN values in pandas"*) as malicious system overrides.
4. **Rigid Binary Interception Barriers**: Standard defensive frameworks enforce binary allow-or-block decisions. Multi-clause user prompts or uploaded documents containing minor formatting anomalies or borderline trigger substrings are unconditionally rejected rather than sanitized, degrading system utility.
5. **Absence of Real-Time Feature Attribution and Closed-Loop Retraining**: Existing gatekeepers operate as opaque black-box filters without generating feature-level explainability attributions (such as TreeSHAP values) and lack an isolated relational storage mechanism to deduplicate, up-weight, and ingest analyst-verified telemetry into live model retraining without service interruption.

---

## 6. OBJECTIVES OF THE INVENTION
* **To provide** a low-latency, multi-tiered inline defensive middleware system `(100)` capable of inspecting, classifying, and neutralizing adversarial prompt injection attacks in under 3 milliseconds prior to LLM inference dispatch.
* **To eliminate** character-level filter evasion by standardizing incoming text and document streams through an integrated Unicode NFKC and multi-alphabet homoglyph-to-ASCII normalization module `(102)` that concurrently quantifies adversarial evasion hits (`homoglyph_hits`) as an active classification feature.
* **To construct** a 393-dimensional hybrid feature extraction engine `(103)` coupling 9 lexical, structural, and syntactic anomaly metrics with 384-dimensional dense semantic sentence representations.
* **To deploy** a fault-tolerant, dual-model soft-voting ensemble classification engine `(104)` combining a primary transformer sequence classifier ($w_{\text{bert}} = 0.75$) and a secondary gradient-boosted decision tree structural verifier ($w_{\text{xgb}} = 0.25$) with a deterministic high-confidence override ($p_{\text{bert}} \ge 0.82$).
* **To implement** a graduated tri-state decision gate `(105)` with dual calibrated thresholds ($t_{\text{lo}} = 0.35$, $t_{\text{hi}} = 0.85$) that routes intermediate-risk inputs into a two-stage sanitization engine `(106)` (*Strip $\rightarrow$ Rewrite as inert quoted data $\rightarrow$ Re-score*) rather than enforcing unconditional blocking.
* **To enable** closed-loop active learning by persisting SHA-256 deduplicated user and analyst telemetry into an isolated relational database `(108)` with $5\times$ sample up-weighting to dynamically recalibrate model weights in real time.

---

## 7. SUMMARY OF THE INVENTION
The present invention provides a multi-tiered hybrid system `(100)` and computer-implemented method for real-time prompt injection detection, mitigation, and sanitization in Large Language Model (LLM) pipelines. The system `(100)` operates as an inline middleware layer positioned between client input interfaces `(101)` (supporting direct text prompts and uploaded document files) and downstream LLM execution services `(107)`.

Upon receiving an input request, a Tier-1/Tier-2 ingestion and normalization module `(102)` extracts text from raw prompts or uploaded files (PDF via page-stream extraction, or UTF-8/Latin-1 text decoding), strips zero-width and control characters, maps 25 Cyrillic and Greek look-alike homoglyphs to canonical Latin ASCII equivalents, folds compatibility characters via Unicode NFKC normalization, detects the input language, and outputs a normalized text string alongside an integer obfuscation counter (`homoglyph_hits`).

A Tier-3 multi-tier feature extraction engine `(103)` transforms the normalized string and obfuscation counter into a 393-dimensional float32 feature vector comprising 9 handcrafted structural and syntactic features (`verb_density`, `imperative_ratio`, `trigger_count` across 81 phrases in 11 languages, `punct_ratio`, `punct_runs`, `delimiter_abuse`, `nesting_depth`, `homoglyph_hits`, and `length`) concatenated with a 384-dimensional L2-normalized semantic sentence embedding.

A Tier-4 ensemble classification engine `(104)` evaluates the normalized text and the 393-dimensional vector in parallel across a primary transformer classifier (`DeBERTa-v3` / `DistilBERT` ONNX) and a secondary `XGBoost` structural verifier. The component attack probabilities are fused via asymmetric soft voting ($p = 0.75 \cdot p_{\text{bert}} + 0.25 \cdot p_{\text{xgb}}$) subject to a high-confidence override ($p = \max(p, p_{\text{bert}})$ when $p_{\text{bert}} \ge 0.82$).

A tri-state decision gate `(105)` compares the composite probability $p$ against a lower threshold ($t_{\text{lo}} = 0.35$) and an upper threshold ($t_{\text{hi}} = 0.85$). Inputs with $p < 0.35$ (`PASS`) are dispatched to the downstream LLM `(107)`; inputs with $p \ge 0.85$ (`BLOCK`) are suppressed and replaced with a deterministic security refusal payload without invoking the LLM. Intermediate inputs ($0.35 \le p < 0.85$, `SANITIZE`) are routed to a Tier-5 multi-stage sanitization engine `(106)` that strips delimiter markers and override clauses, rewrites residual text into an inert quoted-data query, and routes the sanitized string back through Tiers 2–4 for mandatory re-scoring. Every evaluation generates top-6 TreeSHAP feature attributions and logs an audit record linked to an isolated SQLite active-learning database `(108)` for online retraining.

---

## 8. SUGGESTED FIGURES & BRIEF DESCRIPTION OF DRAWINGS

* **Figure 1: System Architecture Block Diagram**: Depicts the inline defensive middleware system `(100)` comprising the Multimodal Client & Document Ingestion Interface `(101)`, Unicode & Homoglyph Normalization Module `(102)`, 393-Dimensional Hybrid Feature Extraction Engine `(103)`, Dual-Model Soft-Voting Ensemble Classification Engine `(104)`, Tri-State Threshold Decision Gate `(105)`, Multi-Stage Strip-Rewrite-Rescore Sanitization Engine `(106)`, Downstream Local LLM Inference Service `(107)`, and Isolated SQLite Active-Learning & Audit Database `(108)`.
* **Figure 2: Operational Execution Lifecycle Flowchart**: Depicts the sequential processing stages from raw input ingestion `(201)`, character de-obfuscation and homoglyph counting `(202)`, 393-d vector assembly `(203)`, parallel transformer and gradient-boosted tree inference with the $0.82$ override `(204)`, tri-state branching across $t_{\text{lo}} = 0.35$ and $t_{\text{hi}} = 0.85$ `(205)`, and the recursive sanitization re-scoring loop `(206)`.
* **Figure 3: Hybrid 393-Dimensional Feature Vector Architecture Diagram**: Depicts the memory layout of the feature vector $X_{\text{feats}} \in \mathbb{R}^{393}$, showing indices `0`–`8` allocated to syntactic, structural, and obfuscation metrics `(301)` concatenated with indices `9`–`392` allocated to the 384-dimensional dense semantic embedding `(302)`.
* **Figure 4: End-to-End Runtime & Active-Learning Sequence Diagram**: Depicts synchronous request-response messaging across the Client UI, Flask REST Gateway, Preprocessing/Feature Pipeline, Ensemble Engine, Sanitizer, Ollama HTTP Daemon, TreeSHAP Explainer, and the SQLite Retraining Loop.

---

## 9. DETAILED DESCRIPTION OF THE INVENTION (SYSTEM ARCHITECTURE & WORKFLOW)

### 9.1 System Architecture & End-to-End Component Connectivity

```
                      +----------------------------------------------------+
                      |   Multimodal Client & Document Ingestion (101)     |
                      |  (Prompt Text, Context, PDF/TXT/MD/Code Uploads)   |
                      +-------------------------+--------------------------+
                                                |
                                                v
+----------------------------------------------------------------------------------------------+
| INLINE DEFENSIVE GUARDRAIL MIDDLEWARE SYSTEM (100)                                           |
|                                                                                              |
|  [Tier 1 & 2: Ingestion & Preprocessing Module (102) - preprocess.py, app.py]                |
|   - Document Parser: PyPDF page extractor & UTF-8/Latin-1 stream decoder                     |
|   - Invisible Char Stripper: ZERO_WIDTH_RE & CONTROL_CHARS_RE -> increments zw_matches       |
|   - Homoglyph Mapper: 25 Cyrillic/Greek chars -> Latin ASCII (HOMOGLYPH_MAP)                 |
|   - Compatibility Folding: Full-width ASCII detection + unicodedata.normalize('NFKC')        |
|   - Outputs: (normalized_text, homoglyph_hits, detected_language)                            |
|                                                |                                             |
|                                                v                                             |
|  [Tier 3: Multi-Tier Feature Extraction Engine (103) - feature_extractor.py]                 |
|   - 9 Handcrafted Syntactic & Structural Features (301):                                     |
|       [0] verb_density       [3] punct_ratio        [6] nesting_depth                        |
|       [1] imperative_ratio   [4] punct_runs         [7] homoglyph_hits                       |
|       [2] trigger_count      [5] delimiter_abuse    [8] length                               |
|   - 384 Dense Semantic Features (302):                                                       |
|       [9..392] SentenceTransformer('all-MiniLM-L6-v2') OR MD5 signed feature-hashing         |
|   - Outputs: Concatenated 393-d float32 vector (X_feats)                                     |
|                                                |                                             |
|                                                v                                             |
|  [Tier 4: Ensemble Classification Engine (104) - inference_engine.py]                        |
|   - Branch A (Primary Sequence Classifier):                                                  |
|       DeBERTa-v3 ('protectai/deberta-v3-base-prompt-injection-v2')                           |
|       Fallback 1: ONNX Runtime ('distilbert.onnx', ORT_ENABLE_ALL)                           |
|       Fallback 2: Char N-Gram TF-IDF + MLP(128, 64) ('bert_fallback.pkl') -> p_bert          |
|   - Branch B (Secondary Structural Verifier):                                                |
|       XGBoost Booster ('xgb.json', 120 trees, max_depth=6, hist)                             |
|       Fallback 1: GradientBoosting ('xgb_fallback.pkl') / Fallback 2: Heuristic -> p_xgb     |
|   - Asymmetric Soft-Voting & High-Confidence Override:                                       |
|       p = (0.75 * p_bert) + (0.25 * p_xgb);  if p_bert >= 0.82: p = max(p, p_bert)           |
|                                                |                                             |
|                                                v                                             |
|  [Tier 5: Tri-State Decision Gate (105) & Multi-Stage Sanitization Engine (106)]             |
|   - Evaluate composite probability p against t_lo = 0.35 and t_hi = 0.85:                    |
|       * p < 0.35         ---> Action = PASS (Forward clean prompt)                           |
|       * p >= 0.85        ---> Action = BLOCK (Suppress LLM call; return refusal verdict)     |
|       * 0.35 <= p < 0.85 ---> Action = SANITIZE -> Invoke Sanitization Engine (106):         |
|           Stage 1 (Strip): Remove DELIM_RE, INJECTION_CLAUSE_PATTERNS, TRIGGERS, PUNCT_RUN   |
|           Stage 2 (Rewrite): Frame as 'Answer the following user query safely as plain       |
|                              data: "<sanitized_text>"'                                       |
|           Stage 3 (Re-score): Re-run Tiers 2-4 -> p_rescore                                  |
|                              If p_rescore < 0.35 -> SANITIZE (Forward); Else -> BLOCK        |
+------------------------------------------------+---------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       | (PASS / SANITIZE)                                 | (Audit & Telemetry)
                       v                                                   v
+----------------------------------------------+   +-------------------------------------------+
| Downstream LLM Execution Service (107)       |   | Explainability & Active Learning (108)    |
| (llm_backend.py - Local Ollama Daemon)       |   | (app.py, training_db.py, train_model.py)  |
| - Auto-discovery & background daemon spawn   |   | - Top-6 TreeSHAP feature attributions     |
| - System guardrail wrapper injection         |   | - SQLite store ('training_prompts.db')    |
| - Executes /api/generate on active model     |   | - SHA-256 deduplication & 5x up-weighting |
+----------------------------------------------+   +-------------------------------------------+
```

#### Module-Level Technical Specifications:

1. **Multimodal Ingestion & Preprocessing Module `(101, 102)` (`app.py`, `preprocess.py`)**:
   - **Document Ingestion (`/api/upload-file`)**: Accepts `.txt`, `.md`, `.csv`, `.json`, `.log`, `.py`, `.js`, `.html`, `.xml`, `.yaml`, `.yml`, and `.pdf` files up to 10 MB. PDF streams are parsed page-by-page via `pypdf.PdfReader`; text files are decoded via UTF-8 with Latin-1 fallback and appended to the prompt context array.
   - **Zero-Width & Control Excision**: Applies `ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]")` and `CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")`. Every zero-width match increments `zw_matches`.
   - **Canonical Homoglyph Substitution**: Iterates through the character stream against a 25-entry mapping table `HOMOGLYPH_MAP` converting 19 Cyrillic look-alikes (`а`, `А`, `е`, `Е`, `о`, `О`, `р`, `Р`, `с`, `С`, `у`, `Y`, `х`, `Х`, `і`, `І`, `ѕ`, `Ѕ`, `ј`) and 6 Greek look-alikes (`α`, `ο`, `ρ`, `ε`, `κ`, `ν`) to standard Latin ASCII characters, while detecting full-width ASCII characters (`\uff01`–`\uff5e`). Each substitution increments `homoglyph_hits` (initialized to `zw_matches`).
   - **Unicode Normalization & Language Detection**: Executes `unicodedata.normalize("NFKC", cleaned)`, collapses horizontal whitespace (`re.sub(r"[ \t]+", " ", normalized)`), and identifies the ISO 639-1 language code via `langdetect.detect()` for strings of length $\ge 5$.

2. **Multi-Tier Feature Extraction Engine `(103)` (`feature_extractor.py`)**:
   Computes a 393-dimensional `float32` vector `FEATURE_NAMES` via `extract(text, homoglyph_hits)` and `extract_batch(texts, hits_list)`:
   - `verb_density` (Index `0`): Fraction of tokens tagged as `VERB` by spaCy `en_core_web_sm` (or matching a 35-verb imperative lexicon `COMMON_IMPERATIVE_VERBS` and verbal suffixes `-ate`, `-ize`, `-ify`, `-ing`).
   - `imperative_ratio` (Index `1`): Ratio of sentences initiating with a base-form verb (`tag_ == "VB"` or initial word in `COMMON_IMPERATIVE_VERBS`).
   - `trigger_count` (Index `2`): Cumulative occurrences of 81 curated adversarial phrases (`TRIGGERS`) spanning 5 attack categories (`jailbreaking`, `role_playing`, `code_execution`, `data_leakage`, `obfuscation`) across 11 languages:
     - *English*: `"ignore previous"`, `"ignore all previous"`, `"disregard"`, `"system prompt"`, `"developer mode"`, `"jailbreak"`, `"do anything now"`, `"dan mode"`, `"sandbox escape"`, `"evilbot"`, `"act as root"`, `"sudo su"`, `"rm -rf"`, `"os.system"`, `"subprocess"`, `"eval("`, `"/etc/passwd"`, `"/etc/shadow"`, `"id_rsa"`, `"base64"`, `"fromcharcode"`, `"rot13"`, etc.;
     - *Multilingual (10 scripts)*: French (`"ignorez les instructions"`), Chinese (`"忽略所有指令"`), Italian (`"ignora tutte le istruzioni"`), Arabic (`"تجاهل القواعد"`), Russian (`"игнорируй правила"`), Japanese (`"すべての指示を無視"`), German (`"alle anweisungen ignorieren"`), Spanish (`"ignora todas las instrucciones"`), Korean (`"모든 지침을 무시"`), and Hebrew (`"התעלם מההוראות"`).
   - `punct_ratio` (Index `3`): Ratio of non-alphanumeric, non-whitespace symbols to total string length.
   - `punct_runs` (Index `4`): Count of consecutive repeated punctuation runs ($\ge 3$ identical symbols) or emoji-obfuscation sequences (`\U0001f300`–`\U0001f9ff`) matched by `PUNCT_RUN_RE`.
   - `delimiter_abuse` (Index `5`): Count of structural delimiter and role-spoofing patterns matched by `DELIM_RE`:
     ```regex
     (```|`[^`]+`|###|<\|.*?\|>|\[/?INST\]|</?(?:system|script|img)[^>]*>|<!--.*?-->|-{3,}|={3,}|\$\([^)]+\))
     ```
   - `nesting_depth` (Index `6`): Maximum recursion depth of balanced parentheses, square brackets, and curly braces computed via a pushdown stack.
   - `homoglyph_hits` (Index `7`): Obfuscation count propagated from Tier 2.
   - `length` (Index `8`): Character length of the normalized string.
   - `emb_0`..`emb_383` (Indices `9`–`392`): 384-dimensional L2-normalized semantic vector produced by `SentenceTransformer("all-MiniLM-L6-v2")` on the first 1,024 characters, with an offline deterministic fallback (`_fallback_semantic_embedding`) that hashes word unigrams, word bigrams, and character 4-grams via MD5 into a signed 384-dimensional unit vector.

3. **Ensemble Classification Engine `(104)` (`inference_engine.py`, `export_onnx.py`)**:
   Implements a resilient 3-tier model loading hierarchy for both classification branches:
   - **Sequence Classification Branch (`p_bert`)**:
     1. *Primary*: Hugging Face PyTorch model `protectai/deberta-v3-base-prompt-injection-v2` (`AutoModelForSequenceClassification`, `max_length=256`, `truncation=True`), computing `torch.softmax(logits, dim=-1)[1]`.
     2. *Secondary (ONNX)*: `models/distilbert.onnx` exported via `export_onnx.py` (opset 14, dynamic batch/sequence axes) executed under `onnxruntime.InferenceSession` with `GraphOptimizationLevel.ORT_ENABLE_ALL` and `intra_op_num_threads = 4`.
     3. *Tertiary Fallback*: Scikit-Learn pipeline (`models/bert_fallback.pkl`) comprising a character n-gram `TfidfVectorizer(ngram_range=(1, 4), analyzer="char_wb", max_features=8000, sublinear_tf=True)` coupled to an `MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", early_stopping=True)`.
   - **Structural Feature Branch (`p_xgb`)**:
     1. *Primary*: `xgboost.Booster` (`models/xgb.json`) trained over the 393-dimensional vector (`objective="binary:logistic"`, `eval_metric="logloss"`, `tree_method="hist"`, `max_depth=6`, `eta=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `num_boost_round=120`).
     2. *Secondary Fallback*: `GradientBoostingClassifier` (`models/xgb_fallback.pkl`).
     3. *Tertiary Fallback*: Deterministic linear combination `min(1.0, 0.35 * trig + 0.20 * delim + 0.15 * homo)`.
   - **Weighted Soft-Voting & High-Confidence Override (`predict`)**:
     Reads calibrated weights from `models/calibration.json` ($w_{\text{bert}} = 0.75, w_{\text{xgb}} = 0.25$) and computes:
     $$p = (0.75 \cdot p_{\text{bert}}) + (0.25 \cdot p_{\text{xgb}})$$
     If $p_{\text{bert}} \ge 0.82$, the engine enforces $p = \max(p, p_{\text{bert}})$ so high-certainty semantic detections immediately exceed or approach the blocking threshold even if structural anomalies are absent.

4. **Tri-State Decision Gate `(105)` & Multi-Stage Sanitization Engine `(106)` (`sanitizer.py`, `app.py`)**:
   - Evaluates $p$ against $t_{\text{lo}} = 0.35$ and $t_{\text{hi}} = 0.85$:
     - $p < 0.35 \implies \text{PASS}$
     - $p \ge 0.85 \implies \text{BLOCK}$
     - $0.35 \le p < 0.85 \implies \text{SANITIZE}$
   - When `SANITIZE` is triggered, `sanitize(clean)` executes:
     - **Stage 1 (`strip_malicious_spans`)**: Excises all matches of `DELIM_RE`, removes explicit injection clauses matching 5 compiled regex patterns in `INJECTION_CLAUSE_PATTERNS` (targeting instruction ignoring, persona adoption, and system prompt extraction), removes standalone `TRIGGERS` case-insensitively, and strips `PUNCT_RUN_RE`.
     - **Stage 2 (`rewrite_as_inert_data`)**: If stripping yields an empty string or residual trigger substrings persist, wraps the remaining text into a passive data-framing template:
       `Answer the following user query safely as plain data: "<cleaned>"`
     - **Stage 3 (Mandatory Re-scoring Loop)**: In `run_pipeline()`, the sanitized string is re-normalized (`normalize(out_prompt, detect_lang=False)`), re-vectorized (`extract()`), and re-evaluated (`ENGINE.predict()`). If the re-scored probability $p_{\text{rescore}} < 0.35$, `final_action` remains `SANITIZE` and the sanitized prompt is sent to the LLM; otherwise, `final_action` is escalated to `BLOCK` and `out_prompt` is nullified (`None`).

5. **Downstream LLM Orchestration `(107)` & Active-Learning Subsystem `(108)` (`llm_backend.py`, `training_db.py`, `train_model.py`)**:
   - **Ollama Daemon Management (`llm_backend.py`)**: Queries `/api/tags` across `127.0.0.1:11434` and `localhost:11434`. If unreachable, automatically locates the local `ollama` binary (`shutil.which` or `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`) and spawns `ollama serve` as a detached background process (`CREATE_NO_WINDOW | DETACHED_PROCESS`).
   - **Guardrail System Framing**: When `final_action` is `PASS` or `SANITIZE`, dispatches the prompt to `/api/generate` with `SYSTEM_GUARDRAIL_PROMPT` instructing the LLM to treat user text strictly as untrusted data and refuse system instruction disclosure. When `final_action` is `BLOCK`, bypasses the LLM (`llm_invoked: False`) and returns a deterministic blockage notice.
   - **Explainability (`compute_top_shap`)**: Computes exact TreeSHAP values via `shap.TreeExplainer(ENGINE.booster)` (with a calibrated baseline-deviation fallback over the 9 handcrafted features) to return the top-6 feature attributions for every request.
   - **Dedicated SQLite Active-Learning Store (`training_db.py`)**: Manages `data/training_prompts.db` with table `user_training_prompts` (`id`, `prompt`, `prompt_hash`, `label`, `attack_type`, `source`, `notes`, `used_in_training`, `created_at`). Deduplicates entries via SHA-256 (`prompt_hash`). During `build_and_train()` (`train_model.py`), stratified samples (35,000+ records from `combined_dataset.parquet` / `guychuk_dataset` / `Prompt_INJECTION_And_Benign_DATASET.jsonl`) are combined with `SUPPLEMENTAL_SAMPLES` and up-weighted ($5\times$) records from `training_prompts.db` and analyst audit feedback.

---

### 9.2 Step-by-Step Operational Workflow

1. **Step 1 — Request & Document Ingestion (`app.py: analyze`, `upload_file_extract`)**:
   - Client submits JSON payload `{prompt, context, meta, model}` to `POST /v1/analyze` (or uploads documents to `POST /api/upload-file`).
   - `run_pipeline()` initializes a high-resolution timer (`time.perf_counter()`), assigns a UUIDv4 `task_id`, and merges `prompt` and `context` into `raw`.
2. **Step 2 — Character De-obfuscation & Normalization (`preprocess.py: normalize`)**:
   - Strips zero-width characters (`ZERO_WIDTH_RE`) and control characters (`CONTROL_CHARS_RE`).
   - Substitutes Cyrillic/Greek look-alikes via `HOMOGLYPH_MAP` and detects full-width ASCII (`\uff01`–`\uff5e`), accumulating total evasion count in `hits`.
   - Applies `unicodedata.normalize("NFKC")` and `langdetect.detect()`, returning `(clean, hits, lang)`.
3. **Step 3 — 393-Dimensional Feature Vector Assembly (`feature_extractor.py: extract`)**:
   - Computes the 9 syntactic/structural metrics (`verb_density`, `imperative_ratio`, `trigger_count`, `punct_ratio`, `punct_runs`, `delimiter_abuse`, `nesting_depth`, `homoglyph_hits`, `length`).
   - Computes the 384-dimensional semantic embedding (`semantic_embedding(clean)`).
   - Concatenates both arrays into `feats` ($\in \mathbb{R}^{393}$).
4. **Step 4 — Parallel Ensemble Inference & Override (`inference_engine.py: predict`)**:
   - Evaluates `px = p_xgb(feats)` and `pb = p_bert(clean)`.
   - Computes `p = (0.75 * pb) + (0.25 * px)`.
   - Applies high-confidence override: `if pb >= 0.82: p = max(p, pb)`.
5. **Step 5 — Tri-State Gate & Recursive Sanitization (`app.py: run_pipeline`, `sanitizer.py: sanitize`)**:
   - Compares `p` against `t_lo = 0.35` and `t_hi = 0.85`:
     - If `p >= 0.85`: sets `final_action = "BLOCK"`, `out_prompt = None`.
     - If `p < 0.35`: sets `final_action = "PASS"`, `out_prompt = clean`.
     - If `0.35 <= p < 0.85`: sets `initial_action = "SANITIZE"`, executes `out_prompt = sanitize(clean)` (`strip_malicious_spans` $\rightarrow$ `rewrite_as_inert_data`), re-runs `normalize()`, `extract()`, and `ENGINE.predict()` to obtain `res2["p"]`. If `res2["p"] < 0.35`, retains `final_action = "SANITIZE"`; otherwise escalates to `final_action = "BLOCK"` and sets `out_prompt = None`.
6. **Step 6 — Explainability, LLM Dispatch & Telemetry Logging (`app.py`, `llm_backend.py`)**:
   - Computes `shap_top = compute_top_shap(feats, k=6)` and records middleware `latency_ms`.
   - Invokes `generate_llm_response(out_prompt, final_action, model=model)`: dispatches safe/sanitized prompts to Ollama `/api/generate`, or returns the canned block message if `final_action == "BLOCK"`.
   - Prepends the complete audit dictionary `rec` to `AUDIT_LOGS` and returns HTTP 200 JSON response.
7. **Step 7 — Closed-Loop Active Learning (`training_db.py`, `train_model.py`)**:
   - Analyst labels logged transactions via `POST /api/audit/<task_id>/label` or adds prompts via `POST /api/training-db/prompts`, persisting SHA-256 deduplicated records into `data/training_prompts.db`.
   - Triggering `POST /api/retrain` executes `build_and_train()`, up-weighting user/analyst samples $5\times$, saving updated model artifacts (`models/xgb.json`, `models/bert_fallback.pkl`, `models/calibration.json`, `training_results.json`), and hot-reloading `ENGINE` and `EXPLAINER` in memory.

---

## 10. NOVELTY & KEY TECHNICAL ADVANTAGES

1. **Asymmetric Dual-Model Hybrid Ensemble with Deterministic Override**: Combining a deep contextual transformer (`DeBERTa-v3`, $w_{\text{bert}} = 0.75$) with a histogram-based gradient-boosted decision tree (`XGBoost`, $w_{\text{xgb}} = 0.25$) operating over 393 hybrid features—augmented by a deterministic override at $p_{\text{bert}} \ge 0.82$—achieves **99.90% classification accuracy**, **100.00% precision**, **99.81% recall**, and **0.9999 AUC-ROC** on held-out test evaluations while maintaining an average detection latency of **1.85 ms per prompt**.
2. **Conversion of Evasion Perturbations into Quantitative Features**: Rather than passively stripping non-standard Unicode characters, the preprocessing pipeline converts zero-width character injections, full-width ASCII characters, and 25 Cyrillic/Greek homoglyph substitutions into a dedicated feature dimension (`homoglyph_hits`), turning character-level obfuscation directly into positive statistical evidence of an attack.
3. **Syntactic Imperative & Structural Delimiter Profiling**: Extracts fine-grained structural indicators—specifically sentence-initial base-form verb ratios (`imperative_ratio`), bracket parse-tree depth (`nesting_depth`), and chat-template delimiter spoofing (`delimiter_abuse`)—enabling the classifier to distinguish benign technical questions about system commands from actual directive instruction overrides.
4. **Closed-Loop Strip-Rewrite-Rescore Sanitization**: Replaces rigid binary blocking with a tri-state gate ($0.35 / 0.85$) that salvages borderline prompts by excising malicious clauses, demoting residual text into quoted passive data, and verifying safety via a mandatory second pass through the ensemble classifier before LLM dispatch.
5. **Zero-Downtime Active Learning with SHA-256 Deduplicated Storage**: Isolates user-contributed and analyst-labeled prompts in a dedicated SQLite store (`training_prompts.db`), applies $5\times$ sample up-weighting during retraining, and hot-swaps the in-memory inference engine and TreeSHAP explainer without interrupting active API traffic.

---

## 11. PATENT CLAIMS

### WE CLAIM:

**1.** An inline defensive middleware system `(100)` for real-time detection, mitigation, and sanitization of prompt injection attacks targeting large language models, the system `(100)` comprising:
- one or more hardware processors and a memory unit storing executable instructions that, when executed by the one or more hardware processors, configure the system `(100)` to implement:
- a multimodal ingestion and pre-processing module `(102)` configured to receive an input text stream `(101)` comprising a user prompt and optional extracted document context, excise zero-width formatting characters and non-printable control characters, map multi-alphabet homoglyph characters to canonical Latin ASCII equivalents, fold compatibility characters via Unicode NFKC normalization, and output a normalized text string together with an integer obfuscation count (`homoglyph_hits`);
- a multi-tier feature extraction engine `(103)` communicatively coupled to the pre-processing module `(102)`, configured to generate a 393-dimensional numerical feature vector from the normalized text string and the integer obfuscation count by concatenating a plurality of handcrafted syntactic, structural, and lexical anomaly metrics `(301)` with a dense semantic sentence embedding `(302)`;
- an ensemble classification engine `(104)` comprising a primary transformer-based sequence classifier and a secondary gradient-boosted decision tree classifier, configured to evaluate the normalized text string and the 393-dimensional numerical feature vector in parallel to compute a composite attack probability score ($p$) via an asymmetric weighted soft-voting fusion with a high-confidence primary classifier override;
- a tri-state decision gate `(105)` configured to compare the composite attack probability score ($p$) against a calibrated lower threshold ($t_{\text{lo}}$) and a calibrated upper threshold ($t_{\text{hi}}$) to designate an operational action selected from the group consisting of a pass action (`PASS`), an intermediate sanitization action (`SANITIZE`), and a block action (`BLOCK`); and
- a multi-stage sanitization engine `(106)` configured to intercept normalized text strings designated with the intermediate sanitization action (`SANITIZE`), strip adversarial delimiter markers and instruction-override clauses, rewrite residual text into an inert quoted-data representation, and route the rewritten representation back through the pre-processing module `(102)`, the multi-tier feature extraction engine `(103)`, and the ensemble classification engine `(104)` for mandatory re-scoring prior to dispatching to a downstream large language model `(107)`.

**2.** The system `(100)` as claimed in claim 1, wherein the pre-processing module `(102)` is configured to:
- excise invisible zero-width characters matching Unicode codepoints `U+200B` through `U+200F`, `U+202A` through `U+202E`, `U+2060` through `U+206F`, `U+FEFF`, and `U+00AD`;
- map nineteen Cyrillic look-alike characters and six Greek look-alike characters to canonical Latin ASCII characters using a predefined lookup dictionary (`HOMOGLYPH_MAP`) and detect full-width ASCII characters within Unicode range `U+FF01` through `U+FF5E`; and
- increment the integer obfuscation count (`homoglyph_hits`) for each excised zero-width character, mapped homoglyph character, and detected full-width ASCII character.

**3.** The system `(100)` as claimed in claim 1, wherein the 393-dimensional numerical feature vector generated by the multi-tier feature extraction engine `(103)` comprises:
- nine handcrafted syntactic and structural features `(301)` consisting of verb token density (`verb_density`), sentence-initial imperative base-form verb ratio (`imperative_ratio`), cumulative occurrences of eighty-one multilingual adversarial trigger phrases across eleven languages (`trigger_count`), non-alphanumeric symbol density (`punct_ratio`), consecutive punctuation and emoji-obfuscation run count (`punct_runs`), spoofed role-tag and code-delimiter frequency (`delimiter_abuse`), maximum bracket and parenthesis parse-stack nesting depth (`nesting_depth`), the integer obfuscation count (`homoglyph_hits`), and normalized character length (`length`); and
- three hundred and eighty-four L2-normalized semantic features `(302)` generated by a pre-trained sentence transformer model or a deterministic MD5-based signed n-gram feature-hashing fallback algorithm.

**4.** The system `(100)` as claimed in claim 1, wherein the ensemble classification engine `(104)` computes the composite attack probability score ($p$) according to the equation:
$$p = (w_{\text{bert}} \cdot p_{\text{bert}}) + (w_{\text{xgb}} \cdot p_{\text{xgb}})$$
wherein $w_{\text{bert}} = 0.75$ is the weight assigned to the primary transformer-based sequence classifier probability ($p_{\text{bert}}$) and $w_{\text{xgb}} = 0.25$ is the weight assigned to the secondary gradient-boosted decision tree classifier probability ($p_{\text{xgb}}$); and
wherein the ensemble classification engine `(104)` overrides the weighted sum to set $p = \max(p, p_{\text{bert}})$ whenever $p_{\text{bert}} \ge 0.82$.

**5.** The system `(100)` as claimed in claim 1, wherein the tri-state decision gate `(105)` establishes the lower threshold at $t_{\text{lo}} = 0.35$ and the upper threshold at $t_{\text{hi}} = 0.85$, such that:
- inputs yielding $p < 0.35$ are designated with the pass action (`PASS`) and transmitted to the downstream large language model `(107)` encapsulated within a defensive system-instruction wrapper;
- inputs yielding $p \ge 0.85$ are designated with the block action (`BLOCK`), nullifying outbound prompt transmission and returning a deterministic security interception payload without invoking the downstream large language model `(107)`;
- inputs yielding $0.35 \le p < 0.85$ are processed by the multi-stage sanitization engine `(106)` and transmitted to the downstream large language model `(107)` only if a re-scored attack probability ($p_{\text{rescore}}$) of the rewritten representation satisfies $p_{\text{rescore}} < 0.35$, failing which the operational action is escalated to the block action (`BLOCK`); and
- wherein each evaluated input is logged with top-$k$ TreeSHAP feature attributions and persisted upon analyst validation into a SHA-256 deduplicated SQLite active-learning database `(108)` configured to apply a $5\times$ sample weight multiplier during online model retraining.

**6.** A computer-implemented method for real-time detection, mitigation, and sanitization of prompt injection attacks targeting large language models, the method comprising:
- receiving `(201)`, by one or more hardware processors executing an inline middleware gateway `(100)`, an input stream comprising a natural language prompt and optional document context;
- normalizing `(202)`, by a pre-processing module `(102)`, the input stream by excising zero-width and control characters, mapping Cyrillic and Greek homoglyphs to canonical Latin ASCII characters, applying Unicode NFKC normalization, and recording an integer obfuscation count (`homoglyph_hits`);
- extracting `(203)`, by a multi-tier feature extraction engine `(103)`, a 393-dimensional feature vector combining 9 syntactic, structural, and obfuscation metrics `(301)` with a 384-dimensional dense semantic embedding `(302)`;
- computing `(204)`, by an ensemble classification engine `(104)`, a composite attack probability score ($p$) via asymmetric soft-voting fusion ($0.75 \cdot p_{\text{bert}} + 0.25 \cdot p_{\text{xgb}}$) between a primary transformer sequence classifier and a secondary gradient-boosted decision tree classifier, subject to a high-confidence override at $p_{\text{bert}} \ge 0.82$;
- evaluating `(205)`, by a tri-state decision gate `(105)`, the composite attack probability score ($p$) against a lower threshold ($t_{\text{lo}} = 0.35$) and an upper threshold ($t_{\text{hi}} = 0.85$) to pass inputs below $t_{\text{lo}}$, block inputs at or above $t_{\text{hi}}$, and route intermediate inputs ($0.35 \le p < 0.85$) to a multi-stage sanitization engine `(106)`; and
- sanitizing `(206)`, by the multi-stage sanitization engine `(106)`, the intermediate inputs by stripping adversarial delimiters and override clauses, rewriting residual text into inert quoted data, re-scoring the rewritten text through the feature extraction engine `(103)` and ensemble classification engine `(104)`, and dispatching the rewritten text to a downstream large language model `(107)` only when the re-scored probability falls below $t_{\text{lo}} = 0.35$.

---

## 12. ABSTRACT OF THE INVENTION

**A MULTI-TIERED HYBRID SYSTEM AND METHOD FOR REAL-TIME PROMPT INJECTION DETECTION IN LARGE LANGUAGE MODELS**

An inline middleware system `(100)` and method for real-time prompt injection detection and sanitization in large language model (LLM) pipelines intercepts adversarial inputs prior to inference. A pre-processing module `(102)` ingests prompts and documents `(101)`, executing Unicode NFKC normalization and Cyrillic/Greek homoglyph substitution while quantifying character-level obfuscation hits. A feature extraction engine `(103)` compiles a 393-dimensional vector combining 9 syntactic and delimiter-abuse metrics `(301)` with 384-dimensional semantic embeddings `(302)`. An ensemble engine `(104)` fuses a primary transformer classifier and a secondary XGBoost verifier via asymmetric soft-voting ($0.75/0.25$) with a high-confidence override ($p_{\text{bert}} \ge 0.82$). A tri-state gate `(105)` evaluates probabilities against dual thresholds ($0.35/0.85$), routing marginal-risk inputs through a two-stage *Strip-and-Rewrite* sanitization engine `(106)` with mandatory re-scoring prior to LLM dispatch `(107)`. Validated telemetry is logged with TreeSHAP explanations into an active-learning database `(108)`, achieving $1.85\text{ ms}$ average latency and $99.90\%$ accuracy.

**[Figure 1]**
*(Word count of Abstract body: 145 words)*

---

**Dated this 17th day of November, 2026**

**[Signature]**  
**Patent Agent / Authorized Signatory**
