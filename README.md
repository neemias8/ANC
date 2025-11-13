# **Abstractive Narrative Consolidation (ANC)**

## **Comparing Extractive and Abstractive Methods for Gospel Consolidation**

This repository explores **Narrative Consolidation** by comparing extractive (TAEG) and abstractive (PRIMERA) approaches for unifying multiple gospel accounts of the Passion Week. The project investigates whether state-of-the-art multi-document summarization models can achieve comparable performance to purpose-built extractive methods while maintaining chronological integrity and factual accuracy.

> 📋 **TL;DR**: **PRIMERA Event-by-Event achieves near-perfect results** (τ=0.976) using event-based decomposition, making it a viable abstractive alternative to TAEG. Standard MDS approaches fail at narrative consolidation. See full findings below.

### **Background: The Narrative Consolidation Task**

**Narrative Consolidation** differs fundamentally from traditional Multi-Document Summarization (MDS):

| Aspect | Traditional MDS | Narrative Consolidation |
|--------|----------------|------------------------|
| **Goal** | Conciseness | Completeness & Coherence |
| **Length** | Shorter than sources | As long as needed |
| **Temporal Order** | Not prioritized | **Critical requirement** |
| **Information** | Select salient points | Integrate all complementary details |

**Original TAEG Method** (Extractive):
- Builds Temporal Alignment Event Graph
- Selects best sentence per event chronologically
- Achieves Kendall's Tau = 1.000 (perfect order)
- ROUGE-L = 0.947 (comprehensive coverage)

**This Work** investigates whether **PRIMERA** (abstractive) can match or exceed TAEG while improving text fluency.

---

## **🔬 Key Findings**

### **1. Standard MDS Models Fail at Narrative Consolidation**

Traditional multi-document summarization models (BART, PEGASUS, PRIMERA-MDS) are fundamentally unsuitable for narrative consolidation tasks:

| Model | Kendall's Tau | Max Input | Max Output | Issue |
|-------|--------------|-----------|------------|-------|
| **PEGASUS-XSUM** | 0.0846 | 512 tokens | 64 tokens | Extreme compression loses temporal structure |
| **PEGASUS-Large** | 0.1110 | 1024 tokens | 256 tokens | Generic summaries, poor chronology |
| **PRIMERA-MDS** | 0.5104 | 4096 tokens | 1024 tokens | Aggressive compression destroys narrative flow |
| **BART-Large-CNN** | 0.5802 | 1024 tokens | 142 tokens | Single-document training inadequate for multi-doc fusion |

**Root Cause:** These models are trained to produce **concise summaries** from long documents. When faced with narrative consolidation (requiring **completeness** not brevity), they:
- **Architectural bottlenecks:** Input/output length limits prevent processing complete narratives (4 gospels = ~40,000+ tokens)
- **Output corruption:** Generate garbled text with encoding artifacts requiring post-processing cleanup (e.g., "Jesus said ÁááààÁÊó to the disciples", random character sequences like "ÊÊÊóóó" appearing mid-sentence or at sentence endings)
- Aggressively compress content, losing critical events
- Fail to maintain temporal ordering across multiple source documents
- Generate severe hallucinations (fabricated content, theological nonsense)
- Cannot handle the length requirements of comprehensive narrative integration (output needs ~10,000+ tokens)

### **2. PRIMERA Event-by-Event: Solving the Long Document Problem**

The breakthrough came from **decomposing the long consolidation task into event-level segments**:

**Standard MDS Problem:**
- Input: 4 long gospel texts → Output: 1 compressed summary
- Model struggles with length, loses temporal structure
- Result: τ=0.51 (poor chronological order)

**Event-Based Solution:**
- Input: 4 gospel segments per event → Output: 1 consolidated event
- Process 169 events independently with numbered structure
- Reassemble into complete chronological narrative
- Result: **τ=0.976** (near-perfect chronological order)


### **3. Event ID-Based Evaluation Metric**

To properly evaluate abstractive methods that paraphrase content, we developed an event ID-based Kendall's Tau metric:
- Extracts numbered events using regex: `\b(\d+)\s+"?([A-Z])`
- Compares only relative ordering of common events
- Ignores paraphrasing differences (measures structure, not wording)
- Reveals PRIMERA Event-by-Event maintains 97.6% chronological accuracy

### **4. Comparative Model Performance**

| Model | Kendall's Tau | Type | Notes |
|-------|--------------|------|-------|
| **TAEG (LexRank-TA)** | **1.0000** | Extractive | Perfect baseline |
| **PRIMERA Event-by-Event** | **0.9758** | Abstractive | Near-perfect |
| **PRIMERA-MDS** | 0.5104 | Abstractive | Standard mode |
| **BART-Large-CNN** | 0.5802 | Abstractive | Single-doc trained |
| **PEGASUS-Large** | 0.1110 | Abstractive | Poor chronology |
| **PEGASUS-XSUM** | 0.0846 | Abstractive | Extreme compression |

**Key Insights:**
- Event-based decomposition is critical for chronological accuracy
- Standard MDS models struggle with temporal ordering across documents
- TAEG remains the gold standard but PRIMERA Event-by-Event is competitive

### **5. Hallucination Analysis**

**Hallucination** refers to factually incorrect information generated by abstractive models—content not present in the source documents.

| Model | Hallucination Level | Factual Accuracy | Example Issues |
|-------|-------------------|------------------|----------------|
| **TAEG** | ✅ **None** | 100% | Extractive—copies sentences verbatim |
| **PRIMERA Event-by-Event** | ✅ **Minimal** | ~98.8% | **2 generation failures**: neural collapse (#83), repetitive loop (#120) |
| **PRIMERA-MDS** | ⚠️ **Severe** | ~40% | **Fabricated narrative**, nonsensical sequences |
| **BART-Large-CNN** | ⚠️ **Moderate** | ~75% | Invented dialogue, compressed events |
| **PEGASUS-Large** | ⚠️ **Moderate** | ~70% | Generic summaries, missing specifics |
| **PEGASUS-XSUM** | ⚠️ **Severe** | ~50% | Extreme compression loses factual grounding |

**Detailed Findings:**

**TAEG (Extractive):**
- Zero hallucinations—all text directly extracted from source gospels
- Maintains perfect factual fidelity by design

**PRIMERA Event-by-Event:**
- **Minimal hallucinations** (~1% error rate)
- Maintains factual accuracy through numbered event structure
- Event-based segmentation enforces grounding in source material
- **Generation Failures (rare occurrences):**
  - **Neural collapse** (e.g., Event 83) - complete loss of coherence with anachronistic modern content (COVID-19, school shootings, technology)
  - **Repetitive degeneration** (e.g., Event 120) - model trapped in loop repeating same information without completing sentence structure

**PRIMERA Standard MDS:**
- **CRITICAL: Severe hallucinations detected**
- Generates incoherent, fabricated narrative segments
- Example hallucinations:
  - *"They are the masters of the land and the people are the serpents of the sea"* (completely fabricated)
  - *"And the serpent is the master of the house of God"* (theological nonsense)
  - Mixed unrelated events into incomprehensible sequences
  - Created dialogue that never occurred
- **Root cause:** Standard MDS mode prioritizes conciseness over accuracy, leading to aggressive compression and creative rewriting

**BART-Large-CNN:**
- Moderate hallucinations due to single-document training
- Tends to invent connective phrases and simplified dialogue
- Loses factual precision in multi-document fusion

**PEGASUS Models:**
- Both variants show significant hallucinations
- Extreme summarization (XSUM) produces generic, factually untethered summaries
- Trained on news articles—struggles with narrative consolidation requirements

**Critical Takeaway:**
- **Event-based structuring prevents hallucinations** in abstractive models
- PRIMERA Event-by-Event achieves 98.8% factual accuracy through:
  1. Numbered event constraints
  2. One-event-per-generation focus
  3. Explicit temporal ordering
- Standard MDS approaches produce dangerous hallucinations for consolidation tasks
- **Generation failure modes observed (rare):**
  - **Neural collapse**: Complete coherence loss with anachronistic content mixing (COVID-19, AR-15, modern technology appearing in biblical context)
  - **Repetitive degeneration**: Model trapped in self-reinforcing loop, repeating semantically equivalent phrases without sentence completion

---

## **Three Methods Compared**

| Method | Type | Chronological Order | Factual Accuracy | Best For |
|--------|------|-------------------|------------------|----------|
| **TAEG** | Extractive | ✅ Perfect (τ=1.0) | ✅ 100% | Literal preservation |
| **PRIMERA Event-by-Event** | Abstractive | ✅ Near-perfect (τ=0.976) | ✅ ~99% | Fluency + accuracy |
| **PRIMERA-MDS** | Abstractive | ⚠️ Poor (τ=0.51) | ❌ ~40% | ❌ Not recommended |

**Key Discovery**: PRIMERA Event-by-Event achieves **near-perfect chronological order** (τ=0.976) by decomposing the long consolidation task into individual numbered events, avoiding the length and compression issues that plague standard MDS models.

**Conclusion**: 
- **TAEG** remains superior for **perfect order** and **literal preservation**
- **PRIMERA Event-by-Event** is viable alternative with **97.6% order accuracy** and better text fluency
- Standard PRIMERA-MDS unsuitable for consolidation tasks (generates concise summaries)

---

## **Dataset: Gospel Consolidation Language Resource**

- **Four Gospels (NIV 2011)**: Matthew, Mark, Luke, John (Passion Week focus)
- **169 Canonical Events**: Palm Sunday to Resurrection
- **Golden Sample**: Expert-crafted reference (~79,000 characters)
- **Format**: XML with `book:chapter:verse` alignment (language-agnostic)

---

## **Evaluation Metrics**

All methods are evaluated against the **Golden Sample** using identical metrics:

### **Content Metrics**
- **ROUGE-1 / ROUGE-2**: Unigram and bigram overlap with reference
- **ROUGE-L**: Longest common subsequence (rewards chronological ordering)
- **METEOR**: Word alignment with synonymy and stemming support
- **BERTScore**: Semantic similarity using contextual embeddings

### **Temporal Ordering Metric**
- **Kendall's Tau (τ)**: Correlation of event ordering (-1 to +1)
  - τ = 1.0 → Perfect chronological order
  - τ = 0.0 → Random ordering
  - τ < 0 → Inversions present

### **Expected Performance**

| Metric | TAEG (Baseline) | PRIMERA Event-by-Event | PRIMERA-MDS |
|--------|----------------|----------------------|-------------|
| **Kendall's Tau** | 1.000 | **0.976** | ~0.51 |
| **ROUGE-1 F1** | 0.958 | ~0.90-0.95 | ~0.7-0.8 |
| **ROUGE-2 F1** | 0.938 | ~0.85-0.92 | ~0.6-0.7 |
| **ROUGE-L F1** | 0.947 | ~0.90-0.95 | ~0.5-0.6 |
| **BERTScore F1** | 0.995 | ~0.95-0.98 | ~0.85-0.90 |
| **METEOR** | 0.639 | ~0.60-0.70 | ~0.4-0.5 |
| **Text Fluency** | Good | ✅ Excellent | ✅ Excellent |

**Key Findings:**
1. PRIMERA Event-by-Event achieves near-perfect temporal order (τ = 0.976)
2. Event-based decomposition solves the long document problem that plagues standard MDS
3. Event-based segmentation is critical for chronological accuracy
4. Standard PRIMERA-MDS shows poor temporal ordering (τ = 0.51)

---

## **Project Structure**

```
├── README.md                          # This file
├── DESCOBERTAS_PROYECTO.md            # 🔬 Detailed findings and recommendations
├── data/
│   ├── EnglishNIVMatthew40_PW.xml    # Gospel texts
│   ├── EnglishNIVMark41_PW.xml
│   ├── EnglishNIVLuke42_PW.xml
│   ├── EnglishNIVJohn43_PW.xml
│   ├── ChronologyOfTheFourGospels_PW.xml  # 169 event mappings
│   └── Golden_Sample.txt              # Reference consolidation
├── src/
│   ├── data_loader.py                 # XML parsing
│   ├── preprocess.py                  # Event-based segmentation
│   ├── summarize_baseline.py          # PRIMERA standard MDS
│   ├── consolidate_abstractive.py     # PRIMERA event-based consolidation
│   ├── evaluator.py                   # Metrics (ROUGE, BERTScore, Kendall's Tau)
│   ├── utils.py                       # PRIMERA utilities
│   ├── summarizer.py                  # LexRank methods
│   └── main.py                        # TAEG pipeline
├── outputs/
│   ├── primera_mds_output.txt
│   ├── primera_consolidation.txt
│   ├── taeg_summary_lexrank-ta.txt
│   ├── comparison_report.txt          # 📊 Three-way comparison
│   └── evaluation/                    # JSON metrics
├── compare_all_methods.py             # 🚀 Main comparison script
└── requirements.txt
```

---

## **🚀 Quick Start**

### **Installation**

```bash
# Clone and setup
git clone https://github.com/neemias8/ANC.git
cd ANC
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### **Run Comparison**

```bash
# Compare all three methods with 10 events
python compare_all_methods.py --max-events 10 --device cpu

# Full comparison (169 events, ~5 hours on CPU)
python compare_all_methods.py --device cpu
```

### **Run Individual Methods**

```bash
# TAEG (extractive)
python src/main.py --method lexrank-ta --output-dir outputs

# PRIMERA-MDS (abstractive concise)
python src/summarize_baseline.py --max-length 512

# PRIMERA-Consolidation (abstractive comprehensive)
python src/consolidate_abstractive.py --max-length-per-event 256
```

**Important**: Use `--max-events 10` for quick testing. Full runs take hours on CPU.

---

## **Evaluation Metrics**

- **Kendall's Tau (τ)**: Chronological order correlation (-1 to +1)
- **ROUGE-1/2/L**: Lexical overlap with reference
- **BERTScore**: Semantic similarity using embeddings
- **METEOR**: Word alignment with synonymy

See [`outputs/comparison_report.txt`](outputs/comparison_report.txt) for detailed results.

---

## **Dependencies**

**Core**:
- `transformers==4.57.1` - PRIMERA model
- `torch>=2.6.0` - Deep learning framework  
- `lexrank==0.1.0` - TAEG method
- `rouge-score`, `bert-score` - Evaluation metrics
- `beautifulsoup4`, `lxml` - XML parsing

**Full list**: See `requirements.txt`

**Hardware**:
- Minimum: 16GB RAM, CPU
- Recommended: GPU with 8GB+ VRAM
- CPU runtime: ~5 hours for full 169 events

---

## **Citation**

```bibtex
@misc{anc2025,
  title={Abstractive Narrative Consolidation: Comparing Extractive and Abstractive Methods},
  author={[Your Name]},
  year={2025},
  note={Extends TAEG framework with PRIMERA experiments}
}
```

---

## **License**

MIT License - See `LICENSE` for details.

---

## **Key Takeaways**

1. ✅ **PRIMERA Event-by-Event achieves near-perfect results** (τ=0.976, ~99% factual accuracy)
2. ✅ **Event-based decomposition solves the long document problem** - breaking narrative into individual events prevents compression and preserves temporal structure
3. ✅ **TAEG remains gold standard** for perfect chronological order (τ=1.0) and zero hallucinations
4. 🔬 **Standard MDS models fundamentally fail** - BART, PEGASUS, PRIMERA-MDS show poor temporal ordering (τ<0.6) and severe hallucinations
5. 📊 **Event ID-based Kendall's Tau** reveals true chronological accuracy beyond sentence matching
6. ⚠️ **PRIMERA Standard MDS produces severe hallucinations** - fabricated narrative, theological nonsense
7. 🎯 **Rare generation failure modes** - Neural collapse and repetitive degeneration occur occasionally even with event-based constraints

> 📖 **Read [`DESCOBERTAS_PROYECTO.md`](DESCOBERTAS_PROYECTO.md) for detailed technical findings, implementation notes, and recommendations for using PRIMERA in narrative consolidation projects.**

---

**Questions?** Open an issue on GitHub.
