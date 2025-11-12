"""
Abstractive Narrative Consolidation using PRIMERA with event-based segmentation.
Maintains chronological order while generating fluent narrative.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import PrimeraPreprocessor
from utils import (
    PrimeraModelLoader,
    set_reproducibility_seed,
    save_output,
    print_generation_config,
    print_gpu_memory
)


class PrimeraAbstractiveConsolidator:
    """
    Abstractive Narrative Consolidation using PRIMERA.
    
    Key features:
    - Event-based segmentation (169 canonical events)
    - Chronological ordering guaranteed by design
    - Abstractive generation for each event
    - Fusion of multiple gospel versions per event
    """
    
    def __init__(
        self,
        model_name: str = "allenai/PRIMERA",
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize PRIMERA abstractive consolidator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            seed: Random seed for reproducibility
        """
        # Set reproducibility
        set_reproducibility_seed(seed)
        
        # Initialize model loader
        print("\n🚀 Initializing PRIMERA Abstractive Consolidator")
        print("="*70)
        self.model_loader = PrimeraModelLoader(
            model_name=model_name,
            device=device
        )
        
        # Initialize preprocessor
        self.preprocessor = PrimeraPreprocessor()
        
        print("✅ Consolidator initialized successfully\n")
    
    def consolidate(
        self,
        max_length_per_event: int = 256,  # MUITO reduzido para evitar alucinações
        min_length_per_event: int = 10,   # Mínimo muito baixo para eventos curtos
        num_beams: int = 4,
        length_penalty: float = 0.8,      # Ligeiramente penaliza textos longos
        no_repeat_ngram_size: int = 3,
        use_event_descriptions: bool = False,  # SEM descrições por padrão
        max_events: Optional[int] = None
    ) -> str:
        """
        Generate abstractive consolidation with event-based segmentation.
        
        Args:
            max_length_per_event: Maximum tokens per event (MUITO conservador)
            min_length_per_event: Minimum tokens per event (permite eventos muito curtos)
            num_beams: Number of beams for beam search
            length_penalty: Length penalty (0.8 = ligeiramente penaliza textos longos)
            no_repeat_ngram_size: Prevents repetition of n-grams
            use_event_descriptions: Include event descriptions in prompts (DESABILITADO)
            max_events: Maximum number of events to process (None = all)
            
        Returns:
            Complete consolidated narrative
        """
        print("\n" + "="*70)
        print("PRIMERA ABSTRACTIVE CONSOLIDATION (Event-Based)")
        print("="*70)
        
        # Prepare event-based inputs
        print("\n📚 Step 1: Preparing event-based inputs...")
        event_inputs = self.preprocessor.prepare_event_based_inputs()
        
        if max_events is not None:
            event_inputs = event_inputs[:max_events]
            print(f"   ⚠️  Limited to first {max_events} events for testing")
        
        print(f"   Total events to process: {len(event_inputs)}")
        
        # Print generation config
        config = {
            "max_length_per_event": max_length_per_event,
            "min_length_per_event": min_length_per_event,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "use_event_descriptions": use_event_descriptions,
            "total_events": len(event_inputs)
        }
        print_generation_config(config)
        
        # Generate for each event
        print("🤖 Step 2: Generating narrative event-by-event...")
        print_gpu_memory()
        
        event_narratives = []
        
        for event_input in tqdm(event_inputs, desc="Processing events"):
            event_id = event_input['event_id']
            description = event_input['description']
            combined_text = event_input['combined_text']  # Já contém <doc-sep> entre evangelhos
            num_gospels = event_input['num_gospels']
            
            # Create prompt for this event
            # IMPORTANTE: combined_text já usa <doc-sep> para separar múltiplos evangelhos
            # SEM PROMPT - apenas o texto dos evangelhos separados por <doc-sep>
            # O modelo PRIMERA foi treinado para reconhecer <doc-sep> como separador
            prompt = combined_text
            
            # Generate narrative for this event
            # Parâmetros MUITO conservadores para evitar alucinações
            try:
                event_narrative = self.model_loader.generate(
                    prompt,
                    max_length=max_length_per_event,
                    min_length=min_length_per_event,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=False,  # Determinístico (beam search puro)
                    repetition_penalty=1.5  # Penaliza fortemente repetições
                )
                
                event_narratives.append(event_narrative)
                
            except Exception as e:
                print(f"\n⚠️  Error processing event {event_id}: {e}")
                # Fallback: use original text
                event_narratives.append(combined_text)
        
        # Combine all event narratives
        print("\n📝 Step 3: Combining event narratives...")
        full_narrative = " ".join(event_narratives)
        
        print(f"✅ Consolidation complete: {len(full_narrative)} characters")
        print(f"   Average per event: {len(full_narrative) / len(event_inputs):.0f} characters")
        
        print(f"\n📝 Preview:\n{full_narrative[:500]}...\n")
        
        print_gpu_memory()
        
        return full_narrative
    
    def consolidate_with_best_gospel_selection(
        self,
        max_events: Optional[int] = None
    ) -> str:
        """
        Alternative approach: Select best gospel per event (no generation).
        This serves as a simpler baseline within PRIMERA experiments.
        
        Args:
            max_events: Maximum number of events to process
            
        Returns:
            Consolidated narrative using best gospel selection
        """
        print("\n" + "="*70)
        print("BEST GOSPEL SELECTION (No Generation)")
        print("="*70)
        
        # Prepare event-based inputs
        event_inputs = self.preprocessor.prepare_event_based_inputs()
        
        if max_events is not None:
            event_inputs = event_inputs[:max_events]
        
        print(f"Processing {len(event_inputs)} events...")
        
        narratives = []
        
        for event_input in event_inputs:
            # Select longest gospel version as "best"
            best_gospel = max(
                event_input['gospel_versions'].items(),
                key=lambda x: len(x[1])
            )
            
            gospel_name, text = best_gospel
            narratives.append(text)
        
        full_narrative = " ".join(narratives)
        
        print(f"✅ Selection complete: {len(full_narrative)} characters")
        
        return full_narrative


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Generate abstractive consolidation using PRIMERA"
    )
    
    parser.add_argument(
        "--model-name",
        default="allenai/PRIMERA",
        help="Hugging Face model name"
    )
    
    parser.add_argument(
        "--max-length-per-event",
        type=int,
        default=2048,
        help="Maximum output length per event in tokens"
    )
    
    parser.add_argument(
        "--min-length-per-event",
        type=int,
        default=50,
        help="Minimum output length per event in tokens"
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
        default=1.5,
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
        default="outputs/primera_consolidation.txt",
        help="Output file path"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process (for testing)"
    )
    
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="Don't use event descriptions in prompts"
    )
    
    parser.add_argument(
        "--best-gospel-only",
        action="store_true",
        help="Use best gospel selection instead of generation"
    )
    
    args = parser.parse_args()
    
    # Convert device
    device = None if args.device == "auto" else args.device
    
    # Initialize consolidator
    consolidator = PrimeraAbstractiveConsolidator(
        model_name=args.model_name,
        device=device,
        seed=args.seed
    )
    
    # Generate consolidation
    if args.best_gospel_only:
        narrative = consolidator.consolidate_with_best_gospel_selection(
            max_events=args.max_events
        )
    else:
        narrative = consolidator.consolidate(
            max_length_per_event=args.max_length_per_event,
            min_length_per_event=args.min_length_per_event,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            use_event_descriptions=not args.no_descriptions,
            max_events=args.max_events
        )
    
    # Save output
    save_output(narrative, args.output, method_name="PRIMERA-Consolidation")
    
    print("\n" + "="*70)
    print("✅ PRIMERA ABSTRACTIVE CONSOLIDATION COMPLETED")
    print("="*70)
    print(f"Output saved to: {args.output}")
    print(f"Output length: {len(narrative)} characters")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
