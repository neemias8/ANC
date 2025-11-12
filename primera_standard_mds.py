#!/usr/bin/env python3
"""
PRIMERA Standard Multi-Document Summarization (MDS)
Single-pass summarization WITHOUT external chronological guidance.
Uses best practices learned from event-based consolidation.
"""

import sys
import re
from pathlib import Path
from typing import List, Optional
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import BiblicalDataLoader


class PRIMERAStandardMDS:
    """PRIMERA standard MDS without chronological guidance."""
    
    def __init__(self, device: str = None):
        """
        Initialize PRIMERA for standard MDS.
        
        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[*] Using device: {self.device}")
        print("[*] Loading PRIMERA model (allenai/PRIMERA)...")
        
        self.model_name = "allenai/PRIMERA"
        self.tokenizer = LEDTokenizer.from_pretrained(self.model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)
        
        print("[OK] Model loaded successfully!")
    
    def _clean_summary(self, text: str) -> str:
        """
        Clean generated summary using best practices learned.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove JSON-like structures
        text = re.sub(r'\{[^}]*\}', '', text)
        
        # Remove metadata patterns
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove CJK characters (Chinese, Japanese, Korean)
        text = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+', '', text)
        
        # Remove specific gibberish patterns
        text = re.sub(r'\benaeoa\b', '', text, flags=re.IGNORECASE)
        
        # Remove 6+ consecutive consonants (likely gibberish)
        text = re.sub(r'\b[bcdfghjklmnpqrstvwxyz]{6,}\b', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _prepare_multi_doc_input(self, gospel_texts: List[str]) -> str:
        """
        Prepare multi-document input using PRIMERA's format.
        
        Args:
            gospel_texts: List of gospel narratives
            
        Returns:
            Formatted input string
        """
        # PRIMERA expects: <doc-sep> between documents
        # Format: "text1 <doc-sep> text2 <doc-sep> text3 <doc-sep> text4"
        return " <doc-sep> ".join(gospel_texts)
    
    def _calculate_adaptive_output_length(
        self,
        gospel_texts: List[str],
        compression_ratio: float = 0.85
    ) -> tuple:
        """
        Calculate adaptive output length based on CONSOLIDATION mindset.
        
        Key insight: We're CONSOLIDATING narratives, not summarizing concisely.
        Use total input across ALL gospels, not just longest.
        
        Args:
            gospel_texts: List of gospel texts
            compression_ratio: Target compression (0.85 = 15% reduction)
            
        Returns:
            (max_tokens, min_tokens)
        """
        # Tokenize each gospel
        token_counts = [
            len(self.tokenizer.encode(text, add_special_tokens=False))
            for text in gospel_texts
        ]
        
        total_tokens = sum(token_counts)
        max_single_gospel = max(token_counts)
        
        # CONSOLIDATION approach: base on TOTAL input
        max_tokens = int(total_tokens * compression_ratio)
        
        # Minimum: ensure we preserve at least 80% of longest gospel
        min_tokens = int(max_single_gospel * 0.80)
        
        # Absolute bounds
        max_tokens = max(30, min(max_tokens, 1024))  # PRIMERA decoder max = 1024
        min_tokens = max(30, min(min_tokens, max_tokens - 10))
        
        print(f"\n Adaptive Length Calculation:")
        print(f"   Total input tokens: {total_tokens:,}")
        print(f"   Longest gospel: {max_single_gospel:,} tokens")
        print(f"   Target max output: {max_tokens:,} tokens ({compression_ratio*100:.0f}% of total)")
        print(f"   Target min output: {min_tokens:,} tokens")
        
        return max_tokens, min_tokens
    
    def summarize_all_gospels(
        self,
        compression_ratio: float = 0.85,
        verbose: bool = True
    ) -> str:
        """
        Generate single consolidated summary from all four gospels.
        
        Uses best practices:
        - Consolidation mindset (completeness over brevity)
        - Adaptive output length (85% of total input)
        - length_penalty=1.0 (neutral, not brevity-focused)
        - Proper generation parameters
        
        Args:
            compression_ratio: Target compression ratio (default 0.85)
            verbose: Print progress
            
        Returns:
            Consolidated narrative
        """
        if verbose:
            print("\n" + "="*80)
            print("PRIMERA Standard MDS - Single-Pass Consolidation")
            print("="*80)
        
        # Load data
        data_loader = BiblicalDataLoader()
        
        # Load all four gospels (already as concatenated text)
        gospels_dict = data_loader.load_all_gospels()
        gospel_texts = [
            gospels_dict["Matthew"],
            gospels_dict["Mark"],
            gospels_dict["Luke"],
            gospels_dict["John"]
        ]
        
        if verbose:
            print(f"\n Loaded {len(gospel_texts)} gospels")
            for i, text in enumerate(gospel_texts, 1):
                print(f"   Gospel {i}: {len(text):,} chars, {len(text.split()):,} words")
        
        # Calculate adaptive output length
        max_output_tokens, min_output_tokens = self._calculate_adaptive_output_length(
            gospel_texts,
            compression_ratio
        )
        
        # Prepare multi-document input
        multi_doc_input = self._prepare_multi_doc_input(gospel_texts)
        
        if verbose:
            print(f"\n Prepared multi-document input: {len(multi_doc_input):,} chars")
        
        # Tokenize with proper LED handling
        # LED/PRIMERA: max_position_embeddings = 4096 (not 16384!)
        max_input_length = min(4096, self.model.config.max_position_embeddings - 1)
        inputs = self.tokenizer(
            multi_doc_input,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        input_length = inputs["input_ids"].shape[1]
        if verbose:
            print(f"   Tokenized: {input_length:,} tokens")
            if input_length >= max_input_length:
                print(f"     Input truncated to {max_input_length} tokens (PRIMERA limit)")
        
        # Check decoder limits
        decoder_max = self.model.config.max_decoder_position_embeddings if hasattr(
            self.model.config, 'max_decoder_position_embeddings'
        ) else 1024
        safe_max_output = min(max_output_tokens, decoder_max - 1)
        safe_min_output = min(min_output_tokens, safe_max_output)
        
        # Generate with CONSOLIDATION parameters
        if verbose:
            print(f"\n Generating consolidated narrative...")
            print(f"   Parameters:")
            print(f"      max_new_tokens: {safe_max_output}")
            print(f"      min_new_tokens: {safe_min_output}")
            print(f"      length_penalty: 1.0 (neutral)")
            print(f"      repetition_penalty: 1.2")
            print(f"      num_beams: 4")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=safe_max_output,  # Output size independent of input
                min_new_tokens=safe_min_output,  # Prevent truncation
                length_penalty=1.0,  # NEUTRAL (not 0.6 for brevity)
                repetition_penalty=1.2,
                num_beams=4,
                early_stopping=False,  # Must respect min_new_tokens
                no_repeat_ngram_size=3
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean
        summary = self._clean_summary(summary)
        
        if verbose:
            print(f"\n Generation complete!")
            print(f"   Output: {len(summary):,} chars, {len(summary.split()):,} words")
            print(f"   Compression: {len(summary)/len(multi_doc_input)*100:.1f}% of input")
        
        return summary


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PRIMERA Standard MDS (single-pass, no chronological guidance)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--compression",
        type=float,
        default=0.85,
        help="Compression ratio (default: 0.85 = 15%% reduction)"
    )
    
    parser.add_argument(
        "--output",
        default="outputs/primera_standard_mds.txt",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    # Initialize
    summarizer = PRIMERAStandardMDS(device=args.device)
    
    # Generate
    summary = summarizer.summarize_all_gospels(
        compression_ratio=args.compression,
        verbose=True
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n Saved to: {output_path}")
    print(f"   Size: {len(summary):,} chars")
    
    # Show preview
    print(f"\n Preview (first 500 chars):")
    print("-" * 80)
    print(summary[:500] + "...")
    print("-" * 80)


if __name__ == "__main__":
    main()
