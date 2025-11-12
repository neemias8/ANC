"""
Preprocessing module for PRIMERA-based narrative consolidation.
Handles event-based segmentation and chronological ordering of gospel texts.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ChronologyLoader, BiblicalDataLoader


class PrimeraPreprocessor:
    """
    Preprocessor for preparing gospel texts for PRIMERA model.
    
    Handles:
    - Event-based segmentation (169 canonical events)
    - Chronological ordering
    - Gospel version grouping for each event
    - Input formatting for PRIMERA
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Directory containing gospel XML files and chronology
        """
        self.chrono_loader = ChronologyLoader(data_dir)
        self.biblical_loader = BiblicalDataLoader(data_dir)
        
        # Load chronology once
        self.events = self.chrono_loader.load_chronology()
        
    def prepare_event_based_inputs(self) -> List[Dict]:
        """
        Prepare inputs organized by events in chronological order.
        
        Returns:
            List of event dictionaries, each containing:
            - event_id: Canonical event ID (1-169)
            - description: Event description
            - gospel_versions: Dict mapping gospel name to text
            - combined_text: All gospel versions concatenated
        """
        event_inputs = []
        
        print(f" Preparing event-based inputs for {len(self.events)} events...")
        
        for event in self.events:
            event_id = event['id']
            description = event['description']
            
            # Extract text from each gospel that mentions this event
            gospel_versions = {}
            
            for gospel in ['matthew', 'mark', 'luke', 'john']:
                reference = event.get(gospel, '') or ''
                reference = reference.strip() if reference else ''
                
                if reference:
                    # Extract specific verses for this gospel/event
                    text = self._extract_verses(gospel, reference)
                    
                    if text:
                        gospel_versions[gospel.capitalize()] = text
            
            # Only include events that have at least one gospel version
            if gospel_versions:
                # Combine versions as separate documents for PRIMERA
                # Use <doc-sep> token to separate multiple gospel accounts
                combined_parts = []
                for gospel_name, text in gospel_versions.items():
                    combined_parts.append(text.strip())
                
                # Join with PRIMERA's multi-document separator
                combined_text = " <doc-sep> ".join(combined_parts)
                
                event_inputs.append({
                    'event_id': event_id,
                    'description': description,
                    'gospel_versions': gospel_versions,
                    'combined_text': combined_text,
                    'num_gospels': len(gospel_versions)
                })
                
                print(f"   Event {event_id:3d}: '{description[:50]}...' ({len(gospel_versions)} gospels, {len(combined_text)} chars)")
        
        print(f"\n Prepared {len(event_inputs)} events with gospel texts")
        return event_inputs
    
    def prepare_mds_input(self) -> str:
        """
        Prepare input for standard MDS (multi-document summarization).
        Returns gospels separated by PRIMERA's document separator.
        
        Returns:
            Multiple documents separated by <doc-sep> token
        """
        print(" Preparing standard MDS input (multiple documents)...")
        
        gospel_texts = self.biblical_loader.load_all_gospels()
        
        # Separate documents with PRIMERA's <doc-sep> token
        # IMPORTANTE: PRIMERA espera mltiplos documentos separados
        documents = []
        for gospel_name, text in gospel_texts.items():
            # Cada evangelho  um documento separado
            documents.append(text.strip())
        
        # Join with <doc-sep> special token for PRIMERA
        combined_text = " <doc-sep> ".join(documents)
        
        print(f" Prepared MDS input: {len(documents)} documents, {len(combined_text)} characters total")
        return combined_text
    
    def _extract_verses(self, gospel: str, reference: str) -> str:
        """
        Extract specific verses from a gospel based on reference.
        
        Args:
            gospel: Gospel name ('matthew', 'mark', 'luke', 'john')
            reference: Reference string like "26:6-13" or "12:1-8"
            
        Returns:
            Extracted verse text or empty string if not found
        """
        try:
            # Parse reference (e.g., "26:6-13" -> chapter 26, verses 6-13)
            if ':' not in reference:
                return ""
            
            chapter_part, verse_part = reference.split(':', 1)
            chapter_num = int(chapter_part)
            
            # Parse verse range
            if '-' in verse_part:
                start_verse, end_verse = verse_part.split('-', 1)
                start_verse = int(start_verse)
                end_verse = int(end_verse)
            else:
                start_verse = end_verse = int(verse_part)
            
            # Load gospel XML
            gospels_map = {
                'matthew': 'EnglishNIVMatthew40_PW.xml',
                'mark': 'EnglishNIVMark41_PW.xml',
                'luke': 'EnglishNIVLuke42_PW.xml',
                'john': 'EnglishNIVJohn43_PW.xml'
            }
            
            xml_file = Path(self.chrono_loader.data_dir) / gospels_map[gospel]
            if not xml_file.exists():
                return ""
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find the specific chapter
            for chapter in root.findall('.//chapter'):
                if chapter.get('number') and int(chapter.get('number')) == chapter_num:
                    verses_text = []
                    
                    # Extract specified verse range
                    for verse in chapter.findall('.//verse'):
                        verse_num = verse.get('number')
                        if verse_num and verse.text:
                            try:
                                v_num = int(verse_num)
                                if start_verse <= v_num <= end_verse:
                                    verses_text.append(verse.text.strip())
                            except ValueError:
                                continue
                    
                    return ' '.join(verses_text)
        
        except Exception as e:
            # Silently fail - just return empty string
            return ""
        
        return ""
    
    def format_for_primera(self, event_input: Dict, task: str = "consolidate") -> str:
        """
        Format an event input for PRIMERA model.
        
        Args:
            event_input: Event dictionary from prepare_event_based_inputs()
            task: Either "consolidate" or "summarize"
            
        Returns:
            Formatted prompt string for PRIMERA
        """
        description = event_input['description']
        combined_text = event_input['combined_text']
        
        if task == "consolidate":
            # For consolidation: emphasize integration of all versions
            prompt = f"Consolidate the following accounts of '{description}' into a single coherent narrative:\n\n{combined_text}"
        elif task == "summarize":
            # For summarization: just summarize
            prompt = f"Summarize the following text:\n\n{combined_text}"
        else:
            # Default: just return the combined text
            prompt = combined_text
        
        return prompt
    
    def get_statistics(self, event_inputs: List[Dict]) -> Dict:
        """
        Get statistics about the prepared inputs.
        
        Args:
            event_inputs: List from prepare_event_based_inputs()
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_events': len(event_inputs),
            'total_chars': sum(len(e['combined_text']) for e in event_inputs),
            'avg_chars_per_event': sum(len(e['combined_text']) for e in event_inputs) / len(event_inputs) if event_inputs else 0,
            'gospel_distribution': {
                'matthew': 0,
                'mark': 0,
                'luke': 0,
                'john': 0
            },
            'multi_gospel_events': 0
        }
        
        for event in event_inputs:
            for gospel in event['gospel_versions'].keys():
                stats['gospel_distribution'][gospel.lower()] += 1
            
            if event['num_gospels'] > 1:
                stats['multi_gospel_events'] += 1
        
        return stats


def main():
    """Test the preprocessor."""
    preprocessor = PrimeraPreprocessor()
    
    # Test event-based preparation
    print("\n" + "="*70)
    print("EVENT-BASED PREPARATION")
    print("="*70)
    event_inputs = preprocessor.prepare_event_based_inputs()
    
    # Show some examples
    print("\n Sample Events:")
    for event in event_inputs[:3]:
        print(f"\nEvent {event['event_id']}: {event['description']}")
        print(f"  Gospels: {list(event['gospel_versions'].keys())}")
        print(f"  Combined length: {len(event['combined_text'])} chars")
        print(f"  Preview: {event['combined_text'][:100]}...")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    stats = preprocessor.get_statistics(event_inputs)
    print(f"Total events: {stats['total_events']}")
    print(f"Total characters: {stats['total_chars']:,}")
    print(f"Average chars/event: {stats['avg_chars_per_event']:.1f}")
    print(f"Multi-gospel events: {stats['multi_gospel_events']}")
    print(f"Gospel distribution:")
    for gospel, count in stats['gospel_distribution'].items():
        print(f"  {gospel.capitalize()}: {count}")
    
    # Test MDS preparation
    print("\n" + "="*70)
    print("MDS PREPARATION")
    print("="*70)
    mds_input = preprocessor.prepare_mds_input()
    print(f"MDS input length: {len(mds_input):,} characters")
    print(f"Preview: {mds_input[:200]}...")


if __name__ == "__main__":
    main()

