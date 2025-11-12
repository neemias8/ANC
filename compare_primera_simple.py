#!/usr/bin/env python3
"""
Simple comparison: Load existing PRIMERA outputs and compare them.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluator import SummarizationEvaluator
from data_loader import BiblicalDataLoader


def main():
    """Compare existing outputs."""
    print("\n" + "="*80)
    print("COMPARISON: PRIMERA STANDARD MDS vs EVENT-BASED CONSOLIDATION")
    print("="*80)
    
    # Load data
    data_loader = BiblicalDataLoader()
    golden_sample = data_loader.load_golden_sample()
    evaluator = SummarizationEvaluator()
    
    # Output files
    mds_file = Path("outputs/primera_standard_mds.txt")
    cons_file = Path("outputs/primera_event_by_event.txt")
    
    if not mds_file.exists():
        print(f"\n[ERROR] {mds_file} not found!")
        return
    
    if not cons_file.exists():
        print(f"\n[ERROR] {cons_file} not found!")
        return
    
    # Load outputs
    print("\n[*] Loading outputs...")
    with open(mds_file, 'r', encoding='utf-8') as f:
        mds_summary = f.read()
    
    with open(cons_file, 'r', encoding='utf-8') as f:
        cons_narrative = f.read()
    
    print(f"  - MDS: {len(mds_summary):,} chars, {len(mds_summary.split()):,} words")
    print(f"  - Consolidation: {len(cons_narrative):,} chars, {len(cons_narrative.split()):,} words")
    
    # Evaluate MDS
    print("\n[*] Evaluating Standard MDS...")
    mds_eval = evaluator.evaluate_summary(
        mds_summary,
        golden_sample,
        is_temporal_anchored=False
    )
    
    # Evaluate Consolidation  
    print("\n[*] Evaluating Event-Based Consolidation...")
    cons_eval = evaluator.evaluate_summary(
        cons_narrative,
        golden_sample,
        is_temporal_anchored=True
    )
    
    # Save evaluations
    eval_dir = Path("outputs/evaluation")
    eval_dir.mkdir(exist_ok=True)
    
    with open(eval_dir / "primera_standard_mds_results.json", 'w') as f:
        json.dump(mds_eval, f, indent=2)
    
    with open(eval_dir / "primera_event_consolidation_results.json", 'w') as f:
        json.dump(cons_eval, f, indent=2)
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print("\n[*] APPROACH")
    print("-" * 80)
    print(f"{'Aspect':<30} | {'Standard MDS':<35} | {'Event Consolidation':<35}")
    print("-" * 80)
    print(f"{'Input Processing':<30} | {'All gospels at once':<35} | {'Event-by-event (169 events)':<35}")
    print(f"{'Chronological Guidance':<30} | {'None (model decides)':<35} | {'External chronology':<35}")
    print(f"{'Model Invocations':<30} | {'1 (single pass)':<35} | {'169 (one per event)':<35}")
    
    print("\n[*] QUANTITATIVE METRICS")
    print("-" * 80)
    print(f"{'Metric':<30} | {'Standard MDS':<20} | {'Event Consolidation':<20} | {'Winner':<15}")
    print("-" * 80)
    
    # Kendall's Tau
    tau_mds = mds_eval["kendall_tau"]
    tau_cons = cons_eval["kendall_tau"]
    winner = "Consolidation" if tau_cons > tau_mds else "MDS" if tau_mds > tau_cons else "Tie"
    print(f"{'Kendall Tau (temporal)':<30} | {tau_mds:>20.4f} | {tau_cons:>20.4f} | {winner:<15}")
    
    # ROUGE scores
    metrics = [
        ("ROUGE-1 F1", ["rouge", "rouge1", "f1"]),
        ("ROUGE-2 F1", ["rouge", "rouge2", "f1"]),
        ("ROUGE-L F1", ["rouge", "rougeL", "f1"]),
    ]
    
    for metric_name, path in metrics:
        val_mds = mds_eval
        val_cons = cons_eval
        for key in path:
            val_mds = val_mds[key]
            val_cons = val_cons[key]
        
        winner = "Consolidation" if val_cons > val_mds else "MDS" if val_mds > val_cons else "Tie"
        print(f"{metric_name:<30} | {val_mds:>20.4f} | {val_cons:>20.4f} | {winner:<15}")
    
    # METEOR
    meteor_mds = mds_eval["meteor"]
    meteor_cons = cons_eval["meteor"]
    winner = "Consolidation" if meteor_cons > meteor_mds else "MDS" if meteor_mds > meteor_cons else "Tie"
    print(f"{'METEOR':<30} | {meteor_mds:>20.4f} | {meteor_cons:>20.4f} | {winner:<15}")
    
    # BERTScore
    bert_mds = mds_eval["bertscore"]["f1"]
    bert_cons = cons_eval["bertscore"]["f1"]
    winner = "Consolidation" if bert_cons > bert_mds else "MDS" if bert_mds > bert_cons else "Tie"
    print(f"{'BERTScore F1':<30} | {bert_mds:>20.4f} | {bert_cons:>20.4f} | {winner:<15}")
    
    print("\n[*] OUTPUT CHARACTERISTICS")
    print("-" * 80)
    print(f"{'Property':<30} | {'Standard MDS':<20} | {'Event Consolidation':<20}")
    print("-" * 80)
    print(f"{'Characters':<30} | {len(mds_summary):>20,} | {len(cons_narrative):>20,}")
    print(f"{'Words':<30} | {len(mds_summary.split()):>20,} | {len(cons_narrative.split()):>20,}")
    
    # Golden sample comparison
    golden_chars = len(golden_sample)
    print(f"{'vs Golden Sample':<30} | {len(mds_summary)/golden_chars*100:>19.1f}% | {len(cons_narrative)/golden_chars*100:>19.1f}%")
    
    print("\n[*] SUMMARY")
    print("-" * 80)
    
    # Count wins
    wins_mds = 0
    wins_cons = 0
    
    if tau_cons > tau_mds:
        wins_cons += 1
        print(f"  * Temporal order: Event Consolidation (tau={tau_cons:.4f} vs {tau_mds:.4f})")
    elif tau_mds > tau_cons:
        wins_mds += 1
        print(f"  * Temporal order: Standard MDS (tau={tau_mds:.4f} vs {tau_cons:.4f})")
    else:
        print(f"  * Temporal order: Tie (tau={tau_cons:.4f})")
    
    # Content coverage (ROUGE-L)
    rouge_l_mds = mds_eval["rouge"]["rougeL"]["f1"]
    rouge_l_cons = cons_eval["rouge"]["rougeL"]["f1"]
    if rouge_l_cons > rouge_l_mds:
        wins_cons += 1
        print(f"  * Content coverage: Event Consolidation (ROUGE-L={rouge_l_cons:.4f} vs {rouge_l_mds:.4f})")
    elif rouge_l_mds > rouge_l_cons:
        wins_mds += 1
        print(f"  * Content coverage: Standard MDS (ROUGE-L={rouge_l_mds:.4f} vs {rouge_l_cons:.4f})")
    
    # Semantic similarity (BERTScore)
    if bert_cons > bert_mds:
        wins_cons += 1
        print(f"  * Semantic similarity: Event Consolidation (BERTScore={bert_cons:.4f} vs {bert_mds:.4f})")
    elif bert_mds > bert_cons:
        wins_mds += 1
        print(f"  * Semantic similarity: Standard MDS (BERTScore={bert_mds:.4f} vs {bert_cons:.4f})")
    
    print(f"\n  Overall: MDS={wins_mds} wins, Consolidation={wins_cons} wins")
    
    print("\n[*] KEY INSIGHTS")
    print("-" * 80)
    print("  Standard MDS:")
    print("    * Single-pass processing (efficient)")
    print("    * Model decides structure autonomously")
    print("    * May struggle with temporal ordering")
    print()
    print("  Event-Based Consolidation:")
    print("    * Guided by external chronology (169 events)")
    print("    * Guarantees temporal preservation")
    print("    * More model invocations (slower but controlled)")
    
    print("\n" + "="*80)
    
    # Save report
    report_file = Path("outputs/primera_comparison_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        # Redirect stdout to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        
        # Re-run printing (simplified)
        print("="*80)
        print("PRIMERA COMPARISON REPORT")
        print("="*80)
        print(f"\nStandard MDS: {len(mds_summary):,} chars")
        print(f"Event Consolidation: {len(cons_narrative):,} chars")
        print(f"\nKendall Tau: MDS={tau_mds:.4f}, Consolidation={tau_cons:.4f}")
        print(f"ROUGE-L F1: MDS={rouge_l_mds:.4f}, Consolidation={rouge_l_cons:.4f}")
        print(f"BERTScore F1: MDS={bert_mds:.4f}, Consolidation={bert_cons:.4f}")
        print(f"METEOR: MDS={meteor_mds:.4f}, Consolidation={meteor_cons:.4f}")
        
        sys.stdout = old_stdout
    
    print(f"\n[*] Report saved to: {report_file}")


if __name__ == "__main__":
    main()
