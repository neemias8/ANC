#!/usr/bin/env python3
"""
Three-way comparison script: TAEG vs PRIMERA-MDS vs PRIMERA-Consolidation
Runs all three methods and generates comparative analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import TAEGPipeline
from summarize_baseline import PrimeraMDSSummarizer
from consolidate_abstractive import PrimeraAbstractiveConsolidator
from evaluator import SummarizationEvaluator
from data_loader import BiblicalDataLoader


class ThreeWayComparison:
    """Compare TAEG, PRIMERA-MDS, and PRIMERA-Consolidation."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize comparison.
        
        Args:
            output_dir: Directory for outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.eval_dir = self.output_dir / "evaluation"
        self.eval_dir.mkdir(exist_ok=True)
        
        # Load golden sample once
        self.data_loader = BiblicalDataLoader()
        self.golden_sample = self.data_loader.load_golden_sample()
        
        # Initialize evaluator
        self.evaluator = SummarizationEvaluator()
        
        self.results = {}
    
    def run_taeg(self) -> Dict[str, Any]:
        """Run TAEG extractive method."""
        print("\n" + "="*80)
        print("METHOD 1: TAEG (Extractive with Temporal Graph)")
        print("="*80)
        
        start_time = time.time()
        
        # Run TAEG pipeline
        pipeline = TAEGPipeline()
        results = pipeline.run_pipeline(
            summary_length=1,
            summarization_method="lexrank-ta"
        )
        
        # Save results
        output_file = self.output_dir / "taeg_summary_lexrank-ta.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results["consolidated_summary"])
        
        execution_time = time.time() - start_time
        
        # Evaluate
        evaluation = self.evaluator.evaluate_summary(
            results["consolidated_summary"],
            self.golden_sample,
            is_temporal_anchored=True
        )
        
        # Save evaluation
        eval_file = self.eval_dir / "taeg_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2)
        
        return {
            "method": "TAEG",
            "output_file": str(output_file),
            "eval_file": str(eval_file),
            "summary": results["consolidated_summary"],
            "evaluation": evaluation,
            "execution_time": execution_time,
            "length_chars": len(results["consolidated_summary"])
        }
    
    def run_primera_mds(
        self,
        max_length: int = 512,
        max_events: int = None,
        device: str = None
    ) -> Dict[str, Any]:
        """Run PRIMERA standard MDS."""
        print("\n" + "="*80)
        print("METHOD 2: PRIMERA-MDS (Abstractive Concise Summary)")
        print("="*80)
        
        start_time = time.time()
        
        # Initialize and run
        summarizer = PrimeraMDSSummarizer(device=device)
        summary = summarizer.summarize(
            max_length=max_length,
            max_events=max_events
        )
        
        # Save results
        output_file = self.output_dir / "primera_mds_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        execution_time = time.time() - start_time
        
        # Evaluate
        evaluation = self.evaluator.evaluate_summary(
            summary,
            self.golden_sample,
            is_temporal_anchored=False
        )
        
        # Save evaluation
        eval_file = self.eval_dir / "primera_mds_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2)
        
        return {
            "method": "PRIMERA-MDS",
            "output_file": str(output_file),
            "eval_file": str(eval_file),
            "summary": summary,
            "evaluation": evaluation,
            "execution_time": execution_time,
            "length_chars": len(summary)
        }
    
    def run_primera_consolidation(
        self,
        max_length_per_event: int = 2048,
        max_events: int = None,
        device: str = None
    ) -> Dict[str, Any]:
        """Run PRIMERA abstractive consolidation."""
        print("\n" + "="*80)
        print("METHOD 3: PRIMERA-Consolidation (Event-Based Abstractive)")
        print("="*80)
        
        start_time = time.time()
        
        # Initialize and run
        consolidator = PrimeraAbstractiveConsolidator(device=device)
        narrative = consolidator.consolidate(
            max_length_per_event=max_length_per_event,
            max_events=max_events
        )
        
        # Save results
        output_file = self.output_dir / "primera_consolidation.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(narrative)
        
        execution_time = time.time() - start_time
        
        # Evaluate
        evaluation = self.evaluator.evaluate_summary(
            narrative,
            self.golden_sample,
            is_temporal_anchored=True  # Event-based should preserve order
        )
        
        # Save evaluation
        eval_file = self.eval_dir / "primera_consolidation_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2)
        
        return {
            "method": "PRIMERA-Consolidation",
            "output_file": str(output_file),
            "eval_file": str(eval_file),
            "summary": narrative,
            "evaluation": evaluation,
            "execution_time": execution_time,
            "length_chars": len(narrative)
        }
    
    def print_comparison_table(self):
        """Print comprehensive comparison table."""
        print("\n" + "="*80)
        print("THREE-WAY METHOD COMPARISON")
        print("="*80)
        
        if len(self.results) < 2:
            print("Not enough results to compare. Run methods first.")
            return
        
        # Extract metrics
        methods = list(self.results.keys())
        
        print("\n📊 METRIC COMPARISON")
        print("-" * 80)
        
        # Header
        print(f"{'Metric':<25} | ", end="")
        for method in methods:
            print(f"{method:>20} | ", end="")
        print("Best Method")
        print("-" * 80)
        
        # Metrics to compare
        metrics = [
            ("Kendall's Tau", "kendall_tau"),
            ("ROUGE-1 F1", ("rouge", "rouge1", "f1")),
            ("ROUGE-2 F1", ("rouge", "rouge2", "f1")),
            ("ROUGE-L F1", ("rouge", "rougeL", "f1")),
            ("BERTScore F1", ("bertscore", "f1")),
            ("METEOR", "meteor"),
        ]
        
        for metric_name, metric_path in metrics:
            print(f"{metric_name:<25} | ", end="")
            
            values = {}
            for method in methods:
                eval_data = self.results[method]["evaluation"]
                
                # Navigate nested dict if necessary
                if isinstance(metric_path, tuple):
                    value = eval_data
                    for key in metric_path:
                        value = value[key]
                else:
                    value = eval_data[metric_path]
                
                values[method] = value
                print(f"{value:>20.3f} | ", end="")
            
            # Find best
            best_method = max(values, key=values.get)
            print(f"{best_method}")
        
        print("-" * 80)
        
        # Additional info
        print(f"\n{'Property':<25} | ", end="")
        for method in methods:
            print(f"{method:>20} | ", end="")
        print()
        print("-" * 80)
        
        # Output length
        print(f"{'Output Length (chars)':<25} | ", end="")
        for method in methods:
            length = self.results[method]["length_chars"]
            print(f"{length:>20,} | ", end="")
        print()
        
        # Execution time
        print(f"{'Execution Time (sec)':<25} | ", end="")
        for method in methods:
            exec_time = self.results[method]["execution_time"]
            print(f"{exec_time:>20.1f} | ", end="")
        print()
        
        print("-" * 80)
        
        # Analysis
        print("\n📝 ANALYSIS:")
        
        # Temporal order
        tau_values = {
            method: self.results[method]["evaluation"]["kendall_tau"]
            for method in methods
        }
        best_tau = max(tau_values, key=tau_values.get)
        print(f"  ✓ Best temporal order: {best_tau} (τ={tau_values[best_tau]:.3f})")
        
        # Content coverage
        rouge_l_values = {
            method: self.results[method]["evaluation"]["rouge"]["rougeL"]["f1"]
            for method in methods
        }
        best_rouge_l = max(rouge_l_values, key=rouge_l_values.get)
        print(f"  ✓ Best content coverage: {best_rouge_l} (ROUGE-L={rouge_l_values[best_rouge_l]:.3f})")
        
        # Semantic similarity
        bert_values = {
            method: self.results[method]["evaluation"]["bertscore"]["f1"]
            for method in methods
        }
        best_bert = max(bert_values, key=bert_values.get)
        print(f"  ✓ Best semantic similarity: {best_bert} (BERTScore={bert_values[best_bert]:.3f})")
        
        print("\n" + "="*80)
    
    def save_comparison_report(self, filename: str = "comparison_report.txt"):
        """Save comparison report to file."""
        report_file = self.output_dir / filename
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # Redirect print to file
            original_stdout = sys.stdout
            sys.stdout = f
            
            self.print_comparison_table()
            
            sys.stdout = original_stdout
        
        print(f"\n💾 Comparison report saved to: {report_file}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare TAEG, PRIMERA-MDS, and PRIMERA-Consolidation"
    )
    
    parser.add_argument(
        "--skip-taeg",
        action="store_true",
        help="Skip TAEG execution (use existing output)"
    )
    
    parser.add_argument(
        "--skip-primera-mds",
        action="store_true",
        help="Skip PRIMERA-MDS execution"
    )
    
    parser.add_argument(
        "--skip-primera-cons",
        action="store_true",
        help="Skip PRIMERA-Consolidation execution"
    )
    
    parser.add_argument(
        "--primera-max-length",
        type=int,
        default=512,
        help="Max length for PRIMERA-MDS"
    )
    
    parser.add_argument(
        "--primera-max-length-per-event",
        type=int,
        default=256,  # MUITO reduzido para evitar alucinações
        help="Max length per event for PRIMERA-Consolidation (tokens)"
    )
    
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Limit number of events for testing"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device for PRIMERA models"
    )
    
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Convert device
    device = None if args.device == "auto" else args.device
    
    # Initialize comparison
    comparison = ThreeWayComparison(output_dir=args.output_dir)
    
    # Run methods
    if not args.skip_taeg:
        comparison.results["TAEG"] = comparison.run_taeg()
    
    if not args.skip_primera_mds:
        comparison.results["PRIMERA-MDS"] = comparison.run_primera_mds(
            max_length=args.primera_max_length,
            max_events=args.max_events,
            device=device
        )
    
    if not args.skip_primera_cons:
        comparison.results["PRIMERA-Consolidation"] = comparison.run_primera_consolidation(
            max_length_per_event=args.primera_max_length_per_event,
            max_events=args.max_events,
            device=device
        )
    
    # Print comparison
    if comparison.results:
        comparison.print_comparison_table()
        comparison.save_comparison_report()
    else:
        print("No methods were run. Use --help for options.")


if __name__ == "__main__":
    main()
