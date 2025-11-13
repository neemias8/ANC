#!/usr/bin/env python3
"""
PRIMERA Event-by-Event Consolidation v3.1

Consolidates gospel narratives event-by-event using PRIMERA correctly
(without prompts). This version maintains the high-quality generation
parameters from v3.1 while integrating with the existing evaluation system.
"""

import sys
import re
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, LEDForConditionalGeneration
import xml.etree.ElementTree as ET

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import ChronologyLoader


class PRIMERAEventConsolidator:
    """
    Consolidates gospel narratives event-by-event using PRIMERA correctly (no prompts).
    Based on v3.1 with proven high-quality results.
    """
    
    def __init__(
        self,
        model_name: str = "allenai/PRIMERA",
        device: str = None,
        data_dir: str = "data"
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.data_dir = Path(data_dir)
        
        print(f"[*] Loading PRIMERA model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[OK] Model loaded successfully")
        
        # Initialize chronology loader
        self.chrono_loader = ChronologyLoader(data_dir=str(self.data_dir))
        
        # Gospel XML files
        self.gospel_files = {
            'matthew': 'EnglishNIVMatthew40_PW.xml',
            'mark': 'EnglishNIVMark41_PW.xml',
            'luke': 'EnglishNIVLuke42_PW.xml',
            'john': 'EnglishNIVJohn43_PW.xml'
        }
    
    def _parse_verse_reference(self, reference: str) -> List[Tuple[int, int, int, Optional[str], Optional[str]]]:
        """
        Parse verse reference (e.g., "26:6-13", "21:18-19a", "14:3-16:4").
        
        Supports:
        - Single verse: "26:6"
        - Verse range: "26:6-13"
        - Half verses: "21:18a", "21:19b"
        - Multi-chapter: "14:3-16:4" (chapter 14 verse 3 to chapter 16 verse 4)
        
        Args:
            reference: Verse reference string
            
        Returns:
            List of (chapter, start_verse, end_verse, start_half, end_half) tuples
        """
        if not reference:
            return []
        
        segments = []
        for segment in re.split(r'[;,]', reference):
            segment = segment.strip()
            if not segment:
                continue
            
            # Try to match multi-chapter reference first (e.g., "14:3-16:4")
            multi_match = re.match(r'(\d+):(\d+[ab]?)-(\d+):(\d+[ab]?)', segment)
            if multi_match:
                start_chapter = int(multi_match.group(1))
                start_verse_str = multi_match.group(2)
                end_chapter = int(multi_match.group(3))
                end_verse_str = multi_match.group(4)
                
                # Parse start verse and half
                if start_verse_str.endswith(('a', 'b')):
                    start_half = start_verse_str[-1]
                    start_verse = int(start_verse_str[:-1])
                else:
                    start_half = None
                    start_verse = int(start_verse_str)
                
                # Parse end verse and half
                if end_verse_str.endswith(('a', 'b')):
                    end_half = end_verse_str[-1]
                    end_verse = int(end_verse_str[:-1])
                else:
                    end_half = None
                    end_verse = int(end_verse_str)
                
                # Add first chapter (from start_verse to end of chapter)
                segments.append((start_chapter, start_verse, 999, start_half, None))  # 999 = to end of chapter
                
                # Add intermediate chapters (all verses)
                for ch in range(start_chapter + 1, end_chapter):
                    segments.append((ch, 1, 999, None, None))
                
                # Add last chapter (from beginning to end_verse)
                segments.append((end_chapter, 1, end_verse, None, end_half))
                
                continue
            
            # Single chapter reference (e.g., "26:6-13" or "26:6")
            match = re.match(r'(\d+):(.+)', segment)
            if not match:
                continue
            
            chapter = int(match.group(1))
            verse_part = match.group(2).strip()
            
            start_half = end_half = None
            if '-' in verse_part:
                start_str, end_str = verse_part.split('-', 1)
                
                start_str = start_str.strip()
                if start_str.endswith(('a', 'b')):
                    start_half = start_str[-1]
                    start_verse = int(start_str[:-1])
                else:
                    start_verse = int(start_str)
                
                end_str = end_str.strip()
                if end_str.endswith(('a', 'b')):
                    end_half = end_str[-1]
                    end_verse = int(end_str[:-1])
                else:
                    end_verse = int(end_str)
            else:
                verse_part = verse_part.strip()
                if verse_part.endswith(('a', 'b')):
                    start_half = end_half = verse_part[-1]
                    start_verse = end_verse = int(verse_part[:-1])
                else:
                    start_verse = end_verse = int(verse_part)
            
            segments.append((chapter, start_verse, end_verse, start_half, end_half))
        
        return segments
    
    def _split_verse_half(self, text: str, half: str) -> str:
        """
        Split a verse into halves (a/b) using intelligent punctuation-based splitting.
        
        Priority:
        1. Strong punctuation (. ! ?) - complete sentences
        2. Medium punctuation (; :) - clause boundaries
        3. Weak punctuation (,) - phrase boundaries
        4. Fallback: split at midpoint
        
        Args:
            text: Verse text
            half: 'a' or 'b'
            
        Returns:
            Requested half of the verse
        """
        # Try strong punctuation first (sentence boundaries)
        for delimiter in ['. ', '! ', '? ']:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                if len(parts) == 2:
                    if half == 'a':
                        return parts[0] + delimiter.strip()
                    else:
                        # Capitalize first letter of second part
                        second_part = parts[1].strip()
                        if second_part:
                            second_part = second_part[0].upper() + second_part[1:]
                        return second_part
        
        # Try medium punctuation (clause boundaries)
        for delimiter in ['; ', ': ']:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                if len(parts) == 2:
                    if half == 'a':
                        return parts[0] + delimiter.strip()
                    else:
                        return parts[1].strip()
        
        # Try weak punctuation (comma) - but only if text is long enough
        if len(text) > 80 and ', ' in text:
            # Find comma closest to midpoint
            mid = len(text) // 2
            commas = [i for i, char in enumerate(text) if char == ',']
            if commas:
                closest_comma = min(commas, key=lambda x: abs(x - mid))
                if half == 'a':
                    return text[:closest_comma + 1].strip()
                else:
                    return text[closest_comma + 1:].strip()
        
        # Fallback: split at word boundary near midpoint
        mid = len(text) // 2
        space_idx = text.rfind(' ', mid - 20, mid + 20)  # Look within 20 chars of midpoint
        if space_idx != -1:
            if half == 'a':
                return text[:space_idx].strip()
            else:
                return text[space_idx:].strip()
        
        # Last resort: return full text (don't split)
        return text

    def extract_gospel_text(self, gospel: str, reference: str) -> str:
        """
        Extract text from a gospel based on verse reference.
        
        Args:
            gospel: Gospel name (lowercase: 'matthew', 'mark', 'luke', 'john')
            reference: Verse reference (e.g., "26:6-13")
            
        Returns:
            Extracted text
        """
        if not reference or gospel not in self.gospel_files:
            return ""
        
        xml_file = self.data_dir / self.gospel_files[gospel]
        if not xml_file.exists():
            return ""
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        segments = self._parse_verse_reference(reference)
        all_verses = []
        
        for chapter_num, start_verse, end_verse, start_half, end_half in segments:
            for chapter in root.findall('.//chapter'):
                if int(chapter.get('number', 0)) == chapter_num:
                    for verse in chapter.findall('.//verse'):
                        verse_num = int(verse.get('number', 0))
                        verse_text = verse.text.strip() if verse.text else ""
                        
                        if not verse_text:
                            continue
                        
                        if start_verse <= verse_num <= end_verse:
                            if verse_num == start_verse and start_half:
                                verse_text = self._split_verse_half(verse_text, start_half)
                            elif verse_num == end_verse and end_half:
                                verse_text = self._split_verse_half(verse_text, end_half)
                            
                            all_verses.append(verse_text)
                    break
        
        return ' '.join(all_verses)

    def extract_event_texts(self, event: Dict) -> Dict[str, str]:
        """
        Extract gospel texts for a specific event.
        
        Args:
            event: Event dictionary with gospel references
            
        Returns:
            Dictionary mapping gospel names to their texts
        """
        gospel_texts = {}
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            reference = event.get(gospel, '') or ''  # Handle None values
            reference = reference.strip()
            if reference:
                text = self.extract_gospel_text(gospel, reference)
                if text:
                    gospel_texts[gospel.capitalize()] = text
        return gospel_texts

    def generate_consolidation(self, multi_doc_input: str) -> str:
        """
        Generates a consolidated narrative from concatenated gospel texts.
        NO PROMPT IS USED - just documents separated by <doc-sep>.
        
        This uses the proven v3.1 parameters for high-quality output.
        
        Args:
            multi_doc_input: Gospel texts joined with <doc-sep>
            
        Returns:
            Generated consolidated text
        """
        inputs = self.tokenizer(
            multi_doc_input,
            return_tensors="pt",
            max_length=4096,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=1024,
                num_beams=5,  # Increased beams for better quality
                length_penalty=1.5,  # Slightly encourage longer, more complete text
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def clean_output(self, text: str) -> str:
        """
        Clean generated output with v3.1 punctuation fix.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove separator token
        text = re.sub(r'<doc-sep>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add space after punctuation if missing (v3.1 FIX)
        # Fixes: "word.Word" -> "word. Word"
        text = re.sub(r'([.!?;:,])([A-Z])', r'\1 \2', text)
        
        # Ensure first letter is capitalized
        if text:
            text = text[0].upper() + text[1:]
        
        return text

    def validate_output(self, generated: str) -> Tuple[bool, List[str]]:
        """
        Validate generated output for quality issues.
        
        Args:
            generated: Generated text
            
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        if len(generated.split()) < 10:
            issues.append("Output too short (< 10 words)")
        
        # Check for garbled text (many non-ascii/latin chars)
        non_latin_chars = re.findall(r'[^\x00-\x7F\u00C0-\u017F]+', generated)
        if len(non_latin_chars) > 10:
            issues.append(f"Contains garbled text: {''.join(non_latin_chars)[:20]}...")

        is_valid = len(issues) == 0
        return is_valid, issues

    def consolidate_event(
        self,
        event: Dict,
        event_num: int,
        total_events: int,
        verbose: bool = True
    ) -> Tuple[str, bool]:
        """
        Consolidate a single event from multiple gospel accounts.
        
        Args:
            event: Event dictionary
            event_num: Current event number
            total_events: Total number of events
            verbose: Print progress
            
        Returns:
            (consolidated_text, success)
        """
        gospel_texts = self.extract_event_texts(event)
        
        if not gospel_texts:
            if verbose:
                print(f"  [{event_num}/{total_events}] No gospel texts found - skipping")
            return "", False

        if len(gospel_texts) == 1:
            if verbose:
                print(f"  [{event_num}/{total_events}] {event['description']} (single gospel, no consolidation needed)")
            return self.clean_output(list(gospel_texts.values())[0]), True

        if verbose:
            gospels_list = ', '.join(gospel_texts.keys())
            print(f"  [{event_num}/{total_events}] {event['description']}")
            print(f"      Consolidating {len(gospel_texts)} gospels: {gospels_list}")

        # Correct input format for PRIMERA: no prompts, just docs separated by <doc-sep>
        multi_doc_input = " <doc-sep> ".join(gospel_texts.values())

        try:
            generated = self.generate_consolidation(multi_doc_input)
            cleaned = self.clean_output(generated)
            is_valid, issues = self.validate_output(cleaned)
            
            if not is_valid:
                if verbose: 
                    print(f"      ⚠ Validation failed: {'; '.join(issues)}")
                    print(f"      → Falling back to longest gospel text.")
                cleaned = self.clean_output(max(gospel_texts.values(), key=len))
            
            if verbose:
                print(f"      [OK] Generated {len(cleaned)} characters")
            
            return cleaned, is_valid
            
        except Exception as e:
            if verbose:
                print(f"      ✗ Generation error: {e}")
            return "", False

    def consolidate_all_events(
        self,
        verbose: bool = True,
        save_progress: bool = True
    ) -> str:
        """
        Consolidate all 169 events from the chronology.
        
        Args:
            verbose: Print progress
            save_progress: Save checkpoints every 20 events
            
        Returns:
            Final consolidated narrative
        """
        if verbose:
            print("\n" + "="*80)
            print("PRIMERA Event-by-Event Consolidation v3.1")
            print("="*80)
        
        # Load chronology using ChronologyLoader
        events = self.chrono_loader.load_chronology()
        
        if verbose:
            print(f"\n[*] Loaded {len(events)} events. Processing...\n")
        
        consolidated_parts = []
        success_count = 0
        start_time = time.time()
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        for i, event in enumerate(events, 1):
            consolidated, success = self.consolidate_event(
                event,
                i,
                len(events),
                verbose
            )
            
            if consolidated:
                # Add event number prefix
                consolidated_with_num = f"{i} {consolidated}"
                consolidated_parts.append(consolidated_with_num)
                if success:
                    success_count += 1
            
            # Save checkpoint every 20 events
            if save_progress and i % 20 == 0:
                checkpoint = "\n\n".join(consolidated_parts)
                checkpoint_path = output_dir / f"checkpoint_{i}.txt"
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    f.write(checkpoint)
                
                if verbose:
                    print(f"   [Checkpoint] Saved to {checkpoint_path}")
        
        # Join all parts
        final_narrative = "\n\n".join(consolidated_parts)
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[OK] Consolidation complete!")
            print(f"    Total time: {total_time/60:.1f} minutes")
            print(f"    Events processed: {len(consolidated_parts)}/{len(events)}")
            print(f"    Validation success rate: {success_count}/{len(events)} ({success_count/len(events)*100:.1f}%)")
            print(f"    Output length: {len(final_narrative):,} characters")
            print("="*80)
        
        return final_narrative


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PRIMERA Event-by-Event Consolidation v3.1"
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
    
    print(f"\n[OK] Saved to: {output_path}")
    print(f"    Size: {len(narrative):,} chars")
    
    # Show preview
    print(f"\n{'='*80}")
    print("Preview (first 500 chars):")
    print('-' * 80)
    print(narrative[:500] + "...")
    print('-' * 80)


if __name__ == "__main__":
    main()
