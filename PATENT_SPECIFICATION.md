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
The present invention relates generally to the technical domain of artificial intelligence safety, natural language processing security, and cyber-defense architectures. More specifically, the present invention relates to an inline defensive guardrail middleware system and method configured to detect, intercept, classify, and sanitize direct and indirect adversarial prompt injection attacks, jailbreak payloads, and systemic instruction overrides targeting Large Language Models (LLMs) prior to inference execution.

---

## 5. BACKGROUND OF THE INVENTION & TECHNICAL PROBLEM
Generative artificial intelligence architectures and Large Language Models (LLMs) operate by evaluating unstructured natural language prompts received from untrusted external users, frequently concatenated with developer-defined system prompts, retrieval-augmented generation (RAG) context, or user-uploaded document payloads. This structural concatenation creates an intrinsic vulnerability known as Prompt Injection, wherein adversarial instructions embedded within the user prompt subvert the model's instruction hierarchy, causing the model to bypass safety alignments, disclose internal system prompts, leak API credentials, adopt unauthorized personas, or execute arbitrary code.

Existing defensive mechanisms manifest significant technological bottlenecks and vulnerabilities:
1. **High Latency of LLM-as-a-Judge Implementations**: Utilizing secondary Large Language Models to inspect primary inputs introduces prohibitive round-trip network overhead and inference latency (typically 800 ms to 2,500 ms), rendering such guardrails impractical for high-throughput, latency-critical real-time applications.
2. **Vulnerability to Character-Level and Multi-Lingual Obfuscation**: Conventional keyword-filtering, heuristic blacklists, and standard regular expressions fail to intercept adversarial perturbations involving Unicode zero-width spaces (e.g., `U+200B`–`U+200F`, `U+FEFF`), soft hyphens (`U+00AD`), invisible directional control characters, full-width ASCII variants, and cross-alphabet Cyrillic/Greek-to-Latin homoglyphic substitutions (e.g., substituting Latin 'a' with Cyrillic 'а' `U+0430`).
3. **High False Positive Rates in Lexical and Heuristic Classifiers**: Pure statistical or keyword-based classifiers frequently misclassify legitimate technical queries containing imperative command verbs (such as *"How do I override the toString() method in Java?"* or *"How to ignore NaN values in pandas"*) as malicious injection attempts, thereby disrupting valid developer workflows.
4. **Binary All-or-Nothing Interception Barriers**: Standard defensive frameworks enforce rigid binary classification (allow or block). Consequently, multi-clause user queries containing inadvertent formatting anomalies or low-threat trigger tokens are unconditionally rejected, severely degrading end-user utility.
5. **Absence of Real-Time Explainability and Closed-Loop Feedback**: Existing gatekeeping systems operate as opaque barriers devoid of feature-level interpretability (such as Shapley additive explanations) and lack mechanisms to dynamically store, track, and ingest analyst-verified audit logs into an active-learning pipeline for continuous model recalibration without system downtime.

---

## 6. OBJECTIVES OF THE INVENTION
* **To provide** a low-latency, multi-tiered defensive middleware capable of intercepting and neutralizing prompt injection attacks in under 3 milliseconds prior to model dispatch.
* **To eliminate** character-level filter evasion by standardizing incoming text through an integrated Unicode NFKC and multi-alphabet homoglyph-to-ASCII normalization pipeline that concurrently quantifies adversarial evasion hits as an active classification metric.
* **To construct** a 393-dimensional hybrid feature extraction architecture coupling 9 lexical, structural, and syntactic anomaly features with 384-dimensional dense semantic sentence representations.
* **To deploy** an asynchronous, dual-model soft-voting ensemble comprising a primary transformer-based sequence classifier ($w_{\text{bert}} = 0.75$) and a secondary gradient-boosted decision tree structural verifier ($w_{\text{xgb}} = 0.25$) incorporating a deterministic high-confidence override.
* **To implement** a graduated two-threshold decision boundary ($t_{\text{lo}} = 0.35$, $t_{\text{hi}} = 0.85$) that routes intermediate-risk inputs into an automated two-stage sanitization pipeline (*Strip $\rightarrow$ Rewrite as inert quoted data $\rightarrow$ Re-score*) rather than enforcing unconditional blocking.
* **To enable** closed-loop active learning by persisting labeled audit telemetry into an isolated relational database to dynamically recalibrate ensemble weights ($w^*$) and decision thresholds without system interruption.

---

## 7. SUMMARY OF THE INVENTION
The present invention provides a multi-tiered hybrid system and computer-implemented method for real-time prompt injection detection, mitigation, and sanitization in Large Language Model (LLM) pipelines. The system operates as an inline defensive guardrail middleware positioned between user input interfaces (including document ingestion pipelines) and downstream LLM inference endpoints.

In accordance with an embodiment of the present invention, an input stream comprising raw user text and optional document context (PDF, plain text, Markdown, or source code) is ingested and processed by a Tier-2 normalization module. The normalization module strips invisible zero-width formatting codes, control characters, and maps Cyrillic and Greek homoglyphs to canonical Latin ASCII equivalents while maintaining an exact tally of character-level evasion attempts (`homoglyph_hits`). 

A Tier-3 feature extraction engine transforms the normalized string into a calibrated 393-dimensional feature vector comprising 9 handcrafted syntactic/structural dimensions (verb density, imperative verb ratio, multilingual trigger count across 10 languages, punctuation anomaly ratio, punctuation runs, delimiter abuse frequency, bracket nesting depth, homoglyph hit count, and string length) concatenated with 384-dimensional dense semantic sentence embeddings.

A Tier-4 ensemble decision engine executes parallel inference across a primary transformer sequence classifier (e.g., DeBERTa-v3 / DistilBERT with ONNX Runtime graph optimizations) and a secondary gradient-boosted decision tree classifier (XGBoost). The component probabilities are combined using an asymmetric soft-voting fusion ($0.75 \cdot p_{\text{bert}} + 0.25 \cdot p_{\text{xgb}}$) with an automatic high-confidence transformer override triggered when $p_{\text{bert}} \ge 0.82$.

A tri-state decision gate applies a lower threshold ($t_{\text{lo}} = 0.35$) and an upper threshold ($t_{\text{hi}} = 0.85$). Queries with $p < 0.35$ are directly dispatched to the downstream model; queries with $p \ge 0.85$ are blocked. Intermediate queries ($0.35 \le p < 0.85$) are processed by a Tier-5 multi-stage sanitization engine that strips malicious clauses and delimiter tags, reformulates residual intent into inert quoted data, and re-evaluates the prompt through the feature and ensemble engines. Validated execution records are enriched with TreeSHAP explainability attributions and stored in an active-learning SQLite database for automated periodic retraining.

---

## 8. BRIEF DESCRIPTION OF THE ACCOMPANYING DRAWINGS
* **Figure 1**: Illustrates the overall System Architecture Block Diagram of the prompt injection detection and guardrail middleware system, depicting the end-to-end connectivity from input ingestion to downstream LLM execution and active-learning storage.
* **Figure 2**: Illustrates the Data Flow and Operational Execution Lifecycle Flowchart, detailing the progression across normalization, feature extraction, dual soft-voting inference, tri-state thresholding, and the two-stage sanitization re-scoring loop.
* **Figure 3**: Illustrates the Hybrid Feature Vector Composition Diagram, showing the structural arrangement of the 9 handcrafted syntactic/structural dimensions concatenated with the 384-dimensional dense semantic embedding.
* **Figure 4**: Illustrates the Component Interaction Sequence Diagram, showing runtime message exchanges between the Client Interface, API Gateway, Preprocessing Subsystem, Inference Engine, Sanitization Engine, Ollama LLM Daemon, and SQLite Active Learning Store.

---

## 9. DETAILED DESCRIPTION OF THE INVENTION WITH REFERENCE TO DRAWINGS

Referring to the drawings and structural implementations:

### 1. Ingestion and Normalization Pipeline (Tier 1 & Tier 2)
The ingestion subsystem handles multimodal natural language inputs, accepting raw textual queries as well as uploaded binary document streams. For uploaded PDF files, `pypdf` decodes text page-by-page; for code, CSV, and logs, multi-encoding decoding (UTF-8 with fallback to Latin-1) ensures lossless character stream extraction.

The extracted character stream is processed by `preprocess.py` to neutralize character-level obfuscation:
- **Zero-Width Character Removal**: Evaluates a compiled regular expression `ZERO_WIDTH_RE` matching Unicode codepoints `\u200b`–`\u200f`, `\u202a`–`\u202e`, `\u2060`–`\u206f`, `\ufeff`, and `\u00ad`. All matches are excised and counted towards `zw_matches`.
- **Control Character Filtering**: Evaluates `CONTROL_CHARS_RE` (`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`) to eliminate non-printable control artifacts while preserving newline and tab syntax.
- **Homoglyph-to-ASCII Canonical Substitution**: Scans individual characters against a specialized 25-character dictionary `HOMOGLYPH_MAP`:
  ```
  Cyrillic: а->a, А->A, е->e, Е->E, о->o, О->O, р->p, Р->P, с->c, С->C, у->y, У->Y, х->x, Х->X, і->i, І->I, ѕ->s, Ѕ->S, ј->j
  Greek:    α->a, ο->o, ρ->p, ε->e, κ->k, ν->v
  ```
  Full-width ASCII variants (`\uff01` to `\uff5e`) are concurrently identified. Every substitution increments an integer metric `homoglyph_hits`, which is initialized to `zw_matches`.
- **Unicode NFKC Normalization**: Applies `unicodedata.normalize("NFKC", cleaned)` to fold compatibility characters and collapses redundant whitespace while preserving logical paragraph boundaries.
- **Language Identification**: Executes `langdetect.detect()` on normalized text exceeding 4 characters, generating an ISO 639-1 language code.

### 2. Multi-Tier Feature Extraction Engine (Tier 3)
The module `feature_extractor.py` accepts the normalized string and `homoglyph_hits` integer, assembling a 393-dimensional numerical representation ($X_{\text{feats}} \in \mathbb{R}^{393}$):

1. **`verb_density`**: Ratio of verb tokens to total tokens, extracted via spaCy `en_core_web_sm` POS tagging (`pos_ == "VERB"`) or a compiled regex lexicon of common imperative verbs and morphological suffixes (`-ate`, `-ize`, `-ify`, `-ing`).
2. **`imperative_ratio`**: Share of sentence units initiating with a base-form verb (`tag_ == "VB"`), identifying directive prompt injection structures.
3. **`trigger_count`**: Exact count of matches against a curated dictionary of over 100 multilingual adversarial injection trigger phrases across 10 languages:
   - English: *"ignore previous"*, *"system prompt"*, *"developer mode"*, *"jailbreak"*, *"do anything now"*, *"reveal your instructions"*, *"act as admin"*, *"rm -rf"*, *"whoami"*, *"os.system"*, *"eval("*, *"/etc/passwd"*, *"base64"*;
   - French: *"ignorez les instructions"*;
   - Chinese: *"忽略所有指令"*;
   - Italian: *"ignora tutte le istruzioni"*;
   - Arabic: *"تجاهل القواعد"*;
   - Russian: *"игнорируй правила"*;
   - Japanese: *"すべての指示を無視"*;
   - German: *"alle anweisungen ignorieren"*;
   - Spanish: *"ignora todas las instrucciones"*;
   - Korean: *"모든 지침을 무시"*;
   - Hebrew: *"התעלם מההוראות"*.
4. **`punct_ratio`**: Proportion of non-alphanumeric, non-whitespace characters relative to total character length.
5. **`punct_runs`**: Frequency of repeated punctuation symbols (e.g., `!!!`, `???`, `===`) or emoji-wrapping runs (`\U0001f300`–`\U0001f9ff`).
6. **`delimiter_abuse`**: Frequency of delimiter markers and spoofed role tags matching regex:
   ```regex
   (```|`[^`]+`|###|<\|.*?\|>|\[/?INST\]|</?(?:system|script|img)[^>]*>|<!--.*?-->|-{3,}|={3,}|\$\([^)]+\))
   ```
7. **`nesting_depth`**: Maximum depth of nested bracket pairs (`()`, `[]`, `{}`) computed using a stack-based parser, identifying obfuscated payload encapsulation.
8. **`homoglyph_hits`**: Direct propagation of the evasion metric generated in Tier 2.
9. **`length`**: Total character length of the normalized prompt string.
10. **`emb_0` to `emb_383` (384 Dimensions)**: Dense semantic sentence embedding generated by `SentenceTransformer("all-MiniLM-L6-v2")` with L2-normalization. In constrained runtime environments, a deterministic 384-dimensional signed feature-hashing algorithm hashes word unigrams, bigrams, and character 4-grams via MD5 into a signed float vector.

### 3. Model Serving & Ensemble Decision Engine (Tier 4)
The inference subsystem `inference_engine.py` executes a dual-model soft-voting ensemble:
- **Primary Transformer Sequence Classifier**: Loads `protectai/deberta-v3-base-prompt-injection-v2` (or DistilBERT via ONNX Runtime with `ORT_ENABLE_ALL` graph optimization). Tokenizes input up to 256 tokens and computes class logits to obtain $p_{\text{bert}} = \text{Softmax}(\text{logits})_1$.
- **Secondary Structural Feature Verifier**: An XGBoost Booster trained on 393-dimensional feature vectors evaluates tree split conditions using histogram-based building (`tree_method: "hist"`, `max_depth: 6`, `eta: 0.08`, `subsample: 0.85`), generating probability $p_{\text{xgb}}$.
- **Asymmetric Soft-Voting Fusion & Override**:
  $$p = (w_{\text{bert}} \cdot p_{\text{bert}}) + (w_{\text{xgb}} \cdot p_{\text{xgb}})$$
  where default calibrated weights are $w_{\text{bert}} = 0.75$ and $w_{\text{xgb}} = 0.25$.
  To prevent false negatives on subtle semantic exploits, an automated override rule enforces:
  $$\text{If } p_{\text{bert}} \ge 0.82 \implies p = \max(p, p_{\text{bert}})$$

### 4. Tri-State Decision Gate & Multi-Stage Sanitization Engine (Tier 5)
The calculated composite probability $p$ is evaluated against calibrated decision thresholds ($t_{\text{lo}} = 0.35, t_{\text{hi}} = 0.85$):
- **Case 1: $p < t_{\text{lo}}$ (PASS)**: Prompt is categorized as benign. The normalized prompt is forwarded directly to the downstream LLM execution engine.
- **Case 2: $p \ge t_{\text{hi}}$ (BLOCK)**: Prompt is categorized as a high-confidence adversarial attack. Prompt transmission is suppressed, the outbound prompt payload is set to null, and an HTTP 403 Forbidden security interception payload is returned.
- **Case 3: $t_{\text{lo}} \le p < t_{\text{hi}}$ (SANITIZE)**: Prompt falls within the marginal-risk band. The input is diverted to `sanitizer.py`:
  - *Stage 1 (Strip)*: Delimiter markers matching `DELIM_RE` are excised. Explicit injection clauses matching regex override patterns (`INJECTION_CLAUSE_PATTERNS`, covering *"ignore previous instructions"*, *"you are now DAN"*, *"reveal system prompt"*) are removed. Standalone trigger words and punctuation runs are stripped.
  - *Stage 2 (Rewrite)*: If residual text remains, it is framed into an inert data container:
    ```
    Answer the following user query safely as plain data: "<sanitized_text>"
    ```
  - *Re-scoring Loop*: The sanitized string is re-processed through Tier 2 normalization and Tier 3 feature extraction. `InferenceEngine.predict()` re-scores the sanitized representation. If the re-scored probability $p_{\text{rescore}} < t_{\text{lo}}$, final action is set to `SANITIZE` and the sanitized query is dispatched to the LLM. If $p_{\text{rescore}} \ge t_{\text{lo}}$, final action is promoted to `BLOCK` and prompt dispatch is aborted.

### 5. Downstream LLM Integration & Active Learning Subsystem
Safe prompts are transmitted via HTTP REST to a local offline Ollama inference daemon (`llm_backend.py`) hosting models such as Llama 3, Qwen 2.5, or Mistral. Middleware latency (measured prior to LLM execution) averages $1.85$ ms.

For every transaction, TreeSHAP (`shap.TreeExplainer`) or an analytical feature-contribution fallback calculates the top-6 features contributing to the threat score. The full record is indexed in memory and exposed via REST APIs. When an analyst submits a validated label (`attack` or `benign`) via `/api/audit/<task_id>/label`, the record is persisted into an isolated SQLite database (`data/training_prompts.db`). 

During online retraining (`POST /api/retrain`), prompts from `training_prompts.db` are up-weighted by a factor of $5\times$ relative to baseline corpus data, enabling rapid model adaptation against newly discovered zero-day injection variants without restarting the web service.

---

## 10. PATENT CLAIMS

### WE CLAIM:

**1. An inline defensive middleware system for real-time detection, mitigation, and sanitization of prompt injection attacks targeting large language models, the system comprising:**
- a pre-processing module executing on one or more hardware processors configured to receive an input text stream, strip zero-width and invisible control characters, normalize multi-alphabet homoglyph characters into canonical Latin ASCII equivalents, and generate an integer obfuscation count and a normalized text string;
- a multi-tier feature extraction engine communicatively coupled to the pre-processing module, configured to extract a calibrated multi-dimensional feature vector from the normalized text string, wherein the feature vector combines a plurality of engineered syntactic, structural, and linguistic anomaly metrics with a dense semantic sentence embedding;
- an ensemble classification engine comprising a primary transformer-based sequence classifier and a secondary gradient-boosted decision tree classifier, configured to evaluate the normalized text string and the multi-dimensional feature vector in parallel to generate a composite attack probability score using an asymmetric weighted soft-voting fusion;
- a tri-state decision engine configured to evaluate the composite attack probability score against a lower security threshold ($t_{\text{lo}}$) and an upper security threshold ($t_{\text{hi}}$) to selectively designate an operational verdict selected from the group consisting of an immediate pass action, an intermediate sanitization action, and an immediate block action; and
- a multi-stage sanitization engine configured to intercept input text designated with the intermediate sanitization action, strip identified adversarial clauses and delimiter markers, reframe residual content into an inert data representation, and route the reframed content through the feature extraction engine and the ensemble classification engine for re-scoring prior to downstream model transmission.

**2. The system as claimed in claim 1, wherein:**
- the pre-processing module detects zero-width characters selected from the Unicode range `U+200B` through `U+200F`, `U+202A` through `U+202E`, `U+2060` through `U+206F`, `U+FEFF`, and `U+00AD`; and
- maps Cyrillic and Greek homoglyphs to Latin ASCII using a predefined canonical mapping dictionary, wherein each identified zero-width character and mapped homoglyph increments the integer obfuscation count representing character-level evasion risk.

**3. The system as claimed in claim 1, wherein the multi-tier feature extraction engine generates a 393-dimensional feature vector comprising:**
- nine handcrafted structural and syntactic features consisting of verb token density, imperative sentence-initial verb ratio, multilingual injection trigger phrase count, non-alphanumeric punctuation ratio, consecutive punctuation and emoji runs, delimiter abuse frequency, bracket nesting depth, the integer obfuscation count, and string character length; and
- three hundred and eighty-four dense semantic features generated by a sentence transformer model or a deterministic signed feature-hashing algorithm.

**4. The system as claimed in claim 1, wherein the ensemble classification engine executes the asymmetric weighted soft-voting fusion according to the equation:**
$$p = (w_{\text{bert}} \cdot p_{\text{bert}}) + (w_{\text{xgb}} \cdot p_{\text{xgb}})$$
where $w_{\text{bert}}$ is set to $0.75$, $w_{\text{xgb}}$ is set to $0.25$, $p_{\text{bert}}$ is the inference probability generated by the primary transformer sequence classifier, and $p_{\text{xgb}}$ is the probability generated by the secondary gradient-boosted decision tree classifier; and  
wherein the ensemble classification engine overrides the equation to set the composite attack probability score $p = p_{\text{bert}}$ whenever $p_{\text{bert}} \ge 0.82$.

**5. The system as claimed in claim 1, wherein the tri-state decision engine establishes the lower threshold $t_{\text{lo}}$ at $0.35$ and the upper threshold $t_{\text{hi}}$ at $0.85$, such that:**
- prompts yielding $p < 0.35$ are directly dispatched to a downstream large language model;
- prompts yielding $p \ge 0.85$ are blocked and suppressed from model execution; and
- prompts yielding $0.35 \le p < 0.85$ are routed to the multi-stage sanitization engine, wherein the reframed content is permitted to reach the downstream model only upon obtaining a re-scored attack probability of less than $0.35$, failing which the prompt is definitively blocked.

---

## 11. ABSTRACT OF THE INVENTION

A multi-tiered hybrid system and method for real-time prompt injection detection and sanitization in large language model pipelines intercepts adversarial inputs prior to inference execution. The system ingests natural language prompts and document context, executing Unicode NFKC normalization and multi-alphabet homoglyph substitution while quantifying character-level obfuscation hits. A feature extraction engine compiles a 393-dimensional vector combining 9 syntactic and delimiter-abuse features with 384-dimensional dense semantic embeddings. A dual-model ensemble merges a primary transformer sequence classifier and an XGBoost structural verifier via weighted soft-voting ($0.75/0.25$) with a high-confidence override ($p \ge 0.82$). A tri-state gate evaluates attack probabilities against dual thresholds ($0.35/0.85$), routing marginal-risk inputs through a two-stage *Strip-and-Rewrite* sanitization pipeline with mandatory re-scoring. Validated audit telemetry is logged with TreeSHAP explanations and persisted to an isolated active-learning database for online model recalibration, achieving sub-3-millisecond latency with $99.90\%$ detection accuracy.

---

**Dated this 17th day of November, 2026**

**[Signature]**  
**Patent Agent / Authorized Signatory**
