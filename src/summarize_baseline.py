"""
Standard MDS (Multi-Document Summarization) using PRIMERA.
Generates a concise summary without event-based structure.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import PrimeraPreprocessor
from utils import (
    PrimeraModelLoader,
    set_reproducibility_seed,
    format_input_for_primera,
    save_output,
    print_generation_config,
    print_gpu_memory
)


class PrimeraMDSSummarizer:
    """
    Standard Multi-Document Summarization using PRIMERA.
    
    This approach treats narrative consolidation as traditional MDS:
    - Concatenates all gospel texts
    - Generates a concise summary
    - Does NOT preserve chronological order explicitly
    - Focuses on main events and common elements
    """
    
    def __init__(
        self,
        model_name: str = "allenai/PRIMERA",
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize PRIMERA MDS summarizer.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            seed: Random seed for reproducibility
        """
        # Set reproducibility
        set_reproducibility_seed(seed)
        
        # Initialize model loader
        print("\n🚀 Initializing PRIMERA MDS Summarizer")
        print("="*70)
        self.model_loader = PrimeraModelLoader(
            model_name=model_name,
            device=device
        )
        
        # Initialize preprocessor
        self.preprocessor = PrimeraPreprocessor()
        
        print("✅ Summarizer initialized successfully\n")
    
    def summarize(
        self,
        max_length: int = 512,
        min_length: int = 100,
        num_beams: int = 4,
        length_penalty: float = 0.8,
        no_repeat_ngram_size: int = 3,
        task_prefix: str = "",  # SEM PROMPT - apenas <doc-sep>
        max_events: Optional[int] = None
    ) -> str:
        """
        Generate a concise MDS summary from all gospels.
        
        Args:
            max_length: Maximum output length in tokens
            min_length: Minimum output length in tokens
            num_beams: Number of beams for beam search
            length_penalty: Length penalty (0.8 = ligeiramente penaliza textos longos)
            no_repeat_ngram_size: Prevents repetition of n-grams
            task_prefix: Task description prefix (VAZIO por padrão - sem prompt)
            max_events: If set, only use first N events (for testing)
            
        Returns:
            Generated summary text
        """
        print("\n" + "="*70)
        print("PRIMERA STANDARD MDS SUMMARIZATION")
        print("="*70)
        
        # Prepare input (all gospels concatenated or limited by events)
        print("\n📚 Step 1: Preparing input...")
        
        if max_events is not None:
            print(f"   ⚠️  Limited to first {max_events} events for testing")
            # Use event-based input but concatenate all text
            event_inputs = self.preprocessor.prepare_event_based_inputs()
            event_inputs = event_inputs[:max_events]
            
            # Combine all events into one input
            all_texts = [e['combined_text'] for e in event_inputs]
            input_text = " ".join(all_texts)
            print(f"   Using {len(event_inputs)} events, {len(input_text):,} characters")
        else:
            input_text = self.preprocessor.prepare_mds_input()
        
        # Format for PRIMERA
        formatted_input = format_input_for_primera(
            input_text,
            task_prefix=task_prefix
        )
        
        print(f"   Input length: {len(formatted_input):,} characters")
        
        # Print generation config
        config = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "task_prefix": task_prefix
        }
        print_generation_config(config)
        
        # Generate summary
        print("🤖 Step 2: Generating summary with PRIMERA...")
        print_gpu_memory()
        
        summary = self.model_loader.generate(
            formatted_input,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=False,  # Determinístico (beam search puro)
            repetition_penalty=1.5  # Penaliza fortemente repetições
        )
        
        print(f"✅ Summary generated: {len(summary)} characters")
        print(f"\n📝 Preview:\n{summary[:500]}...\n")
        
        print_gpu_memory()
        
        return summary


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Generate standard MDS summary using PRIMERA"
    )
    
    parser.add_argument(
        "--model-name",
        default="allenai/PRIMERA",
        help="Hugging Face model name"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum output length in tokens"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Minimum output length in tokens"
    )
    
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search"
    )
    
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=2.0,
        help="Length penalty for generation"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run model on"
    )
    
    parser.add_argument(
        "--output",
        default="outputs/primera_mds_output.txt",
        help="Output file path"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Convert device
    device = None if args.device == "auto" else args.device
    
    # Initialize summarizer
    summarizer = PrimeraMDSSummarizer(
        model_name=args.model_name,
        device=device,
        seed=args.seed
    )
    
    # Generate summary
    summary = summarizer.summarize(
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty
    )
    
    # Save output
    save_output(summary, args.output, method_name="PRIMERA-MDS")
    
    print("\n" + "="*70)
    print("✅ PRIMERA MDS SUMMARIZATION COMPLETED")
    print("="*70)
    print(f"Output saved to: {args.output}")
    print(f"Output length: {len(summary)} characters")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
