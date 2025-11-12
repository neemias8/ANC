# **Abstractive Narrative Consolidation (ANC)**

## **Comparing Extractive and Abstractive Methods for Gospel Consolidation**

This repository explores **Narrative Consolidation** by comparing extractive (TAEG) and abstractive (PRIMERA) approaches for unifying multiple gospel accounts of the Passion Week. The project investigates whether state-of-the-art multi-document summarization models can achieve comparable performance to purpose-built extractive methods while maintaining chronological integrity and factual accuracy.

> 📋 **TL;DR**: After extensive experimentation, **TAEG (extractive) outperforms PRIMERA (abstractive)** for this task. See [`DESCOBERTAS_PROYECTO.md`](DESCOBERTAS_PROYECTO.md) for detailed findings on proper PRIMERA usage (`<doc-sep>` format, no prompts, conservative parameters).

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

After extensive experimentation, we discovered:

### **1. TAEG (Extractive) Outperforms PRIMERA (Abstractive)**

| Metric | TAEG ⭐ | PRIMERA-MDS | PRIMERA-Consolidation |
|--------|------|-------------|----------------------|
| **Kendall's Tau** | **1.000** | 0.673 | 0.649 |
| **ROUGE-L F1** | **0.947** | 0.012 | 0.057 |
| **BERTScore F1** | **0.995** | 0.848 | 0.892 |
| **Speed (CPU)** | **35.9s** | 56.0s | 63.7s |

**Conclusion**: For tasks requiring **perfect chronological order** and **literal text preservation** (religious texts, legal documents), extractive methods are superior.

### **2. Critical Discovery: PRIMERA Without Prompts**

**Wrong approach** (causes hallucinations):
```python
prompt = f"Consolidate these {n} accounts: {texts}"  # ❌ Generates hallucinations
```

**Correct approach**:
```python
input_text = f"{doc1} <doc-sep> {doc2} <doc-sep> {doc3}"  # ✅ Zero hallucinations
```

**Why**: PRIMERA was trained on `<doc-sep>` format without explicit prompts. Adding instructions confuses the model.

### **3. Generation Parameters That Work**

```python
# ✅ Recommended
max_length = 256
length_penalty = 0.8
do_sample = False  # Beam search (deterministic)
repetition_penalty = 1.5
# ❌ Do NOT use 'temperature' with beam search (ignored)
```

### **4. PRIMERA-MDS Too Concise**

PRIMERA-MDS generated only **762 characters** for 10 events (compared to 79K for TAEG). It focuses on summarization, not comprehensive consolidation.

> 📖 **See [`DESCOBERTAS_PROYECTO.md`](DESCOBERTAS_PROYECTO.md) for detailed findings, parameter evolution, and practical recommendations.**

---

## **Dataset: Gospel Consolidation Language Resource**

- **Four Gospels (NIV 2011)**: Matthew, Mark, Luke, John (Passion Week focus)
- **169 Canonical Events**: Palm Sunday to Resurrection
- **Golden Sample**: Expert-crafted reference (~79,000 characters)
- **Format**: XML with `book:chapter:verse` alignment (language-agnostic)

---

## **Three Methods Compared**

| Method | Type | Chronological Order | Coverage | Best For |
|--------|------|-------------------|----------|----------|
| **TAEG** | Extractive | ✅ Perfect (τ=1.0) | ✅ Complete | Literal preservation |
| **PRIMERA-MDS** | Abstractive | ⚠️ Partial (τ=0.67) | ❌ Concise only | Brief summaries |
| **PRIMERA-Consolidation** | Abstractive | ⚠️ Good (τ=0.65) | ⚠️ Good | Experimental |

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

| Metric | TAEG (Baseline) | PRIMERA-MDS | PRIMERA-Consolidation (Target) |
|--------|----------------|-------------|-------------------------------|
| **Kendall's Tau** | 1.000 | ~0.3 | **~0.95-1.00** |
| **ROUGE-1 F1** | 0.958 | ~0.7-0.8 | **~0.90-0.95** |
| **ROUGE-2 F1** | 0.938 | ~0.6-0.7 | **~0.85-0.92** |
| **ROUGE-L F1** | 0.947 | ~0.5-0.6 | **~0.90-0.95** |
| **BERTScore F1** | 0.995 | ~0.85-0.90 | **~0.95-0.98** |
| **METEOR** | 0.639 | ~0.4-0.5 | **~0.60-0.70** |
| **Text Fluency** | Good | ✅ Excellent | ✅ Excellent |

**Key Hypotheses:**
1. PRIMERA-Consolidation will achieve near-perfect temporal order (Tau ≈ 1.0)
2. Content coverage will rival TAEG but with better textual fluency
3. Abstractive reformulation may slightly reduce lexical metrics (ROUGE) but maintain semantic equivalence (BERTScore)
4. Standard PRIMERA-MDS will show typical summarization behavior (low Tau, partial coverage)

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

1. ✅ **TAEG (extractive) is superior** for tasks requiring perfect chronological order and literal preservation
2. 🔬 **PRIMERA requires `<doc-sep>` format** - prompts cause hallucinations  
3. ⚙️ **Conservative parameters essential**: `max_length=256`, `length_penalty=0.8`, `repetition_penalty=1.5`
4. ❌ **Don't use `temperature`** with `do_sample=False` (beam search ignores it)
5. 📊 **PRIMERA-MDS too concise** - generates summaries, not consolidations

> 📖 **Read [`DESCOBERTAS_PROYECTO.md`](DESCOBERTAS_PROYECTO.md) for comprehensive findings, parameter evolution, and practical recommendations for using PRIMERA in similar projects.**

---

**Questions?** Open an issue on GitHub.
