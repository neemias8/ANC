#!/usr/bin/env python3
"""
PRIMERA Event-by-Event Consolidation

Similar to TAEG's event-by-event approach, but uses PRIMERA to consolidate
each of the 169 events instead of LexRank. This bypasses the decoder's
1024-token limit by generating one event at a time.
"""

import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import ChronologyLoader, BiblicalDataLoader
from evaluator import SummarizationEvaluator


class PRIMERAEventConsolidator:
    """
    Consolidates gospel events one-by-one using PRIMERA.
    
    Similar to TAEG's approach, but uses abstractive summarization (PRIMERA)
    instead of extractive (LexRank) for each event.
    """
    
    def __init__(self, device: str = None, model_name: str = "allenai/PRIMERA"):
        """
        Initialize the consolidator.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
            model_name: HuggingFace model name
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        
        # Load PRIMERA model
        print(f"[*] Loading PRIMERA model on {self.device}...")
        self.tokenizer = LEDTokenizer.from_pretrained(model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("[✓] Model loaded")
        
        # Data loaders
        self.chrono_loader = ChronologyLoader()
        self.bible_loader = BiblicalDataLoader()
        self.evaluator = SummarizationEvaluator()
    
    def _extract_verses_for_event(self, event: Dict) -> Dict[str, str]:
        """
        Extract the actual verse text for each gospel that mentions this event.
        
        Args:
            event: Event dictionary from ChronologyLoader
            
        Returns:
            Dictionary mapping gospel names to their verse texts
        """
        import xml.etree.ElementTree as ET
        
        gospels_map = {
            'matthew': 'EnglishNIVMatthew40_PW.xml',
            'mark': 'EnglishNIVMark41_PW.xml',
            'luke': 'EnglishNIVLuke42_PW.xml',
            'john': 'EnglishNIVJohn43_PW.xml'
        }
        
        event_texts = {}
        
        for gospel_key in ['matthew', 'mark', 'luke', 'john']:
            reference = event.get(gospel_key, '')
            if not reference or reference is None:
                continue
            reference = reference.strip()
            if not reference:
                continue
            
            try:
                # Parse reference like "26:6-13", "21:18-19a", "21:19b-22", "12:36b"
                parts = reference.split(':')
                if len(parts) != 2:
                    continue
                
                chapter_num = int(parts[0])
                verse_range = parts[1]
                
                # Check for 'a' or 'b' suffix (first/second half of verse)
                # Examples: "18-19a" (verses 18-19, keep only first half of 19)
                #           "19b-22" (verses 19-22, keep only second half of 19)
                #           "36b" (verse 36, only second half)
                start_half = None  # 'a', 'b', or None (full verse)
                end_half = None
                
                # Parse verse range
                if '-' in verse_range:
                    start_part, end_part = verse_range.split('-')
                    
                    # Check start verse for a/b suffix
                    if start_part.endswith('a') or start_part.endswith('b'):
                        start_half = start_part[-1]
                        start_verse = int(start_part[:-1])
                    else:
                        start_verse = int(start_part)
                    
                    # Check end verse for a/b suffix
                    if end_part.endswith('a') or end_part.endswith('b'):
                        end_half = end_part[-1]
                        end_verse = int(end_part[:-1])
                    else:
                        end_verse = int(end_part)
                else:
                    # Single verse, might have a/b suffix
                    if verse_range.endswith('a') or verse_range.endswith('b'):
                        start_half = verse_range[-1]
                        end_half = start_half
                        start_verse = end_verse = int(verse_range[:-1])
                    else:
                        start_verse = end_verse = int(verse_range)
                
                # Load XML and extract verses
                xml_file = Path("data") / gospels_map[gospel_key]
                if not xml_file.exists():
                    continue
                
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Find chapter
                for chapter in root.findall('.//chapter'):
                    if int(chapter.get('number', 0)) == chapter_num:
                        # Extract verses in range
                        verses = []
                        for verse in chapter.findall('.//verse'):
                            verse_num = int(verse.get('number', 0))
                            verse_text = verse.text.strip() if verse.text else ""
                            
                            if not verse_text:
                                continue
                            
                            # Determine if this verse should be included and which half
                            if start_verse <= verse_num <= end_verse:
                                # Check if we need to take only half of the verse
                                if verse_num == start_verse and start_half:
                                    # First verse with a/b suffix
                                    if start_half == 'a':
                                        # Take first half (split at period, semicolon, or middle)
                                        verse_text = self._get_first_half(verse_text)
                                    elif start_half == 'b':
                                        # Take second half
                                        verse_text = self._get_second_half(verse_text)
                                elif verse_num == end_verse and end_half:
                                    # Last verse with a/b suffix
                                    if end_half == 'a':
                                        verse_text = self._get_first_half(verse_text)
                                    elif end_half == 'b':
                                        verse_text = self._get_second_half(verse_text)
                                # else: use full verse
                                
                                verses.append(verse_text)
                        
                        if verses:
                            event_texts[gospel_key.capitalize()] = ' '.join(verses)
                        break
            
            except (ValueError, IndexError, AttributeError) as e:
                # If parsing fails, skip this reference
                continue
        
        return event_texts
    
    def _get_first_half(self, verse_text: str) -> str:
        """
        Get the first half of a verse text.
        Splits at natural breaking points (period, semicolon, colon) or at midpoint.
        
        Args:
            verse_text: Full verse text
            
        Returns:
            First half of the verse
        """
        # Try to split at punctuation marks (period, semicolon, colon)
        for delimiter in ['. ', '; ', ': ']:
            if delimiter in verse_text:
                parts = verse_text.split(delimiter, 1)
                # Return first part with the delimiter
                return parts[0] + delimiter.strip()
        
        # If no natural break point, split at middle
        mid = len(verse_text) // 2
        # Find nearest space to avoid cutting words
        space_idx = verse_text.rfind(' ', 0, mid + 20)
        if space_idx > mid - 20:
            return verse_text[:space_idx].strip()
        
        # Fallback: return first half by character count
        return verse_text[:mid].strip()
    
    def _get_second_half(self, verse_text: str) -> str:
        """
        Get the second half of a verse text.
        Splits at natural breaking points (period, semicolon, colon) or at midpoint.
        
        Args:
            verse_text: Full verse text
            
        Returns:
            Second half of the verse
        """
        # Try to split at punctuation marks
        for delimiter in ['. ', '; ', ': ']:
            if delimiter in verse_text:
                parts = verse_text.split(delimiter, 1)
                if len(parts) > 1:
                    # Return second part
                    return parts[1].strip()
        
        # If no natural break point, split at middle
        mid = len(verse_text) // 2
        # Find nearest space
        space_idx = verse_text.find(' ', mid - 20)
        if space_idx < mid + 20 and space_idx != -1:
            return verse_text[space_idx:].strip()
        
        # Fallback: return second half by character count
        return verse_text[mid:].strip()
    
    def _clean_summary(self, text: str) -> str:
        """
        Clean generated summary by removing metadata, separators, and non-English characters.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text suitable for narrative
        """
        import re
        
        if not text:
            return ''
        
        # Remove document metadata patterns like "Document 1 (Matthew):"
        text = re.sub(r'Document\s*\d*\s*(\([^)]+\))?\s*:+', '', text, flags=re.IGNORECASE)
        
        # Remove lines that are just gospel labels (e.g. "Matthew:")
        text = re.sub(r'^(Matthew|Mark|Luke|John)\s*:\s*', '', text, flags=re.IGNORECASE | re.M)
        
        # Remove separator patterns (bars, equals, dashes)
        text = re.sub(r'[-=_|\[\]]{3,}', ' ', text)
        
        # Remove other common metadata patterns
        text = re.sub(r'\[\d+\]', '', text)  # [1], [2], etc.
        text = re.sub(r'\(\d+:\d+[ab]?-?\d*[ab]?\)', '', text)  # (21:18-19a) references
        
        # Remove CJK and other scripts that appeared as noise (common ranges)
        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]+', '', text)
        
        # Remove URLs and domain patterns
        text = re.sub(r'https?://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # Remove source attribution patterns (more comprehensive)
        text = re.sub(r'Information from:[^.\n]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Source:[^.\n]*', '', text, flags=re.IGNORECASE)
        
        # Remove JSON-like noise (common in web scraping artifacts)
        text = re.sub(r'\{[^}]*\}', '', text)  # Remove JSON objects (any content)
        text = re.sub(r'__\w+__', '', text)  # Remove __property__ patterns
        text = re.sub(r'"[^"]*"\s*:\s*"[^"]*"', '', text)  # Remove "key":"value" patterns
        text = re.sub(r'"[^"]*"\s*:\s*\d+', '', text)  # Remove "key":123 patterns
        
        # Remove specific known garbage patterns
        text = re.sub(r'\benaeoa\b', '', text, flags=re.IGNORECASE)  # Specific gibberish seen
        text = re.sub(r'\bfootnote\s*\d*\b', '', text, flags=re.IGNORECASE)  # Footnote markers
        
        # Remove nonsense word patterns - VERY CONSERVATIVE
        # Only remove words with 6+ consecutive consonants (very rare in real English)
        # This preserves biblical names like "Bethphage" but catches extreme gibberish
        text = re.sub(r'\b\w*[bcdfghjklmnpqrstvwxyz]{6,}\w*\b', '', text, flags=re.IGNORECASE)
        
        # Strip any remaining control characters
        text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF]', '', text)
        
        # Normalize whitespace and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Fix spaces before punctuation
        text = re.sub(r'\.{2,}', '. ', text)  # Ellipses -> single period + space
        
        # Ensure space after sentence-ending punctuation (. ! ?)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space between ". A" -> ". A"
        text = re.sub(r'([.!?])(["\'"])', r'\1 \2', text)  # Add space between ". '" -> ". '"
        
        text = text.strip()
        
        # Capitalize first character only (avoid full sentence re-capitalization)
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _consolidate_event(
        self,
        event: Dict,
        event_num: int,
        total_events: int,
        verbose: bool = True
    ) -> str:
        """
        Consolidate a single event using PRIMERA.
        
        Args:
            event: Event dictionary from ChronologyLoader
            event_num: Current event number (1-indexed)
            total_events: Total number of events
            verbose: Print progress
            
        Returns:
            Consolidated narrative for this event
        """
        # Extract gospel texts for this event
        gospel_texts = self._extract_verses_for_event(event)
        
        if not gospel_texts:
            if verbose:
                print(f"   [!] Event {event_num}/{total_events}: No gospel texts found - skipping")
            return ""
        
        # Prepare multi-document input (PRIMERA format: text1 <doc-sep> text2 <doc-sep> text3)
        multi_doc_input = " <doc-sep> ".join(gospel_texts.values())
        
        # Calculate adaptive output length
        # For event consolidation: aim for 80-90% of total input length
        total_tokens = sum(
            len(self.tokenizer.encode(text, add_special_tokens=False))
            for text in gospel_texts.values()
        )
        
        max_output_tokens = int(total_tokens * 0.85)  # 85% of input
        min_output_tokens = int(total_tokens * 0.70)  # At least 70%
        
        # Clamp to PRIMERA decoder limits (max 1024)
        max_output_tokens = max(30, min(max_output_tokens, 1024))
        min_output_tokens = max(30, min(min_output_tokens, max_output_tokens - 10))
        
        if verbose:
            print(f"   Event {event_num}/{total_events}: {len(gospel_texts)} gospels, "
                  f"{total_tokens} input tokens -> {min_output_tokens}-{max_output_tokens} output tokens")
        
        # Tokenize input (PRIMERA encoder max = 4096)
        max_input_length = min(4096, self.model.config.max_position_embeddings - 1)
        inputs = self.tokenizer(
            multi_doc_input,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate consolidated narrative
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_output_tokens,
                min_new_tokens=min_output_tokens,
                length_penalty=1.0,  # Neutral (not brevity-focused)
                repetition_penalty=1.2,
                num_beams=4,
                early_stopping=False,  # Must respect min_new_tokens
                no_repeat_ngram_size=3
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean
        summary = self._clean_summary(summary)
        
        return summary
    
    def consolidate_all_events(
        self,
        verbose: bool = True,
        save_progress: bool = True
    ) -> str:
        """
        Process all 169 events one-by-one and concatenate results.
        
        Args:
            verbose: Print progress
            save_progress: Save intermediate results every 20 events
            
        Returns:
            Complete consolidated narrative
        """
        if verbose:
            print("\n" + "="*80)
            print("PRIMERA Event-by-Event Consolidation")
            print("="*80)
        
        # Load chronology (169 events)
        events = self.chrono_loader.load_chronology()
        
        if verbose:
            print(f"\n[*] Loaded {len(events)} events from chronology")
            print(f"[*] Processing each event individually...")
        
        consolidated_parts = []
        start_time = time.time()
        
        for i, event in enumerate(events, 1):
            event_summary = self._consolidate_event(
                event,
                event_num=i,
                total_events=len(events),
                verbose=verbose
            )
            
            if event_summary:
                consolidated_parts.append(event_summary)
            
            # Save progress every 20 events
            if save_progress and i % 20 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(events) - i) * avg_time
                
                if verbose:
                    print(f"\n   [Progress] {i}/{len(events)} events processed "
                          f"({i/len(events)*100:.1f}%)")
                    print(f"   [Time] Elapsed: {elapsed/60:.1f}m, "
                          f"Remaining: ~{remaining/60:.1f}m")
                
                # Save checkpoint
                checkpoint = "\n\n".join(consolidated_parts)
                checkpoint_path = Path("outputs") / f"primera_event_checkpoint_{i}.txt"
                checkpoint_path.parent.mkdir(exist_ok=True)
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    f.write(checkpoint)
                
                if verbose:
                    print(f"   [Checkpoint] Saved to {checkpoint_path}")
        
        # Join all parts
        final_narrative = "\n\n".join(consolidated_parts)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n[✓] Consolidation complete!")
            print(f"    Total time: {total_time/60:.1f} minutes")
            print(f"    Output length: {len(final_narrative):,} chars")
            print(f"    Events processed: {len(consolidated_parts)}/{len(events)}")
        
        return final_narrative


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PRIMERA Event-by-Event Consolidation (169 events)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--output",
        default="outputs/primera_event_by_event.txt",
        help="Output file path"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress checkpoints"
    )
    
    args = parser.parse_args()
    
    # Initialize
    consolidator = PRIMERAEventConsolidator(device=args.device)
    
    # Generate
    narrative = consolidator.consolidate_all_events(
        verbose=True,
        save_progress=not args.no_progress
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(narrative)
    
    print(f"\n[✓] Saved to: {output_path}")
    print(f"    Size: {len(narrative):,} chars")
    
    # Evaluate against Golden Sample
    print(f"\n{'='*80}")
    print("Evaluating against Golden Sample")
    print('='*80)
    
    evaluator = SummarizationEvaluator()
    loader = BiblicalDataLoader()
    golden_sample = loader.load_golden_sample()
    
    scores = evaluator.evaluate(narrative, golden_sample)
    
    print(f"\nResults:")
    print(f"  Kendall's Tau:  {scores['kendall_tau']:.4f}")
    print(f"  ROUGE-L:        {scores['rouge_l']:.4f}")
    if scores.get('meteor') is not None:
        print(f"  METEOR:         {scores['meteor']:.4f}")
    print(f"  BERTScore:      {scores['bertscore']:.4f}")
    
    # Show preview
    print(f"\n{'='*80}")
    print("Preview (first 500 chars):")
    print('-' * 80)
    print(narrative[:500] + "...")
    print('-' * 80)


if __name__ == "__main__":
    main()
