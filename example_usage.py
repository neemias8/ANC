#!/usr/bin/env python3
"""
Example Usage of Gospel Consolidator v2

This script demonstrates how to use the GospelConsolidatorV2 class
programmatically in your own code.
"""

from gospel_consolidator_v2 import GospelConsolidatorV2
from pathlib import Path


def example_1_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Initialize consolidator
    consolidator = GospelConsolidatorV2(
        data_dir="data",
        device="cuda"  # or "cpu"
    )
    
    # Process first 5 events
    narrative = consolidator.consolidate_all_events(
        max_events=5,
        verbose=True
    )
    
    # Save output
    output_file = Path("outputs/example1_output.txt")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(narrative)
    
    print(f"\n[✓] Saved to: {output_file}")
    print(f"[✓] Length: {len(narrative)} characters")


def example_2_single_event():
    """Example 2: Process a single specific event."""
    print("\n" + "=" * 80)
    print("Example 2: Single Event Processing")
    print("=" * 80)
    
    consolidator = GospelConsolidatorV2(data_dir="data")
    
    # Load all events
    events = consolidator.load_chronology()
    
    # Process event #1 (Mary anoints Jesus)
    event = events[0]
    print(f"\nProcessing Event #{event['id']}: {event['description']}")
    
    consolidated, success = consolidator.consolidate_event(
        event,
        event_num=1,
        total_events=1,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("Result:")
    print("-" * 80)
    print(consolidated)
    print("-" * 80)
    print(f"Success: {success}")


def example_3_custom_validation():
    """Example 3: Custom validation and fallback handling."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Validation")
    print("=" * 80)
    
    consolidator = GospelConsolidatorV2(data_dir="data")
    
    # Load events
    events = consolidator.load_chronology()[:3]
    
    results = []
    for i, event in enumerate(events, 1):
        consolidated, success = consolidator.consolidate_event(
            event,
            event_num=i,
            total_events=len(events),
            verbose=False
        )
        
        if consolidated:
            results.append({
                'event_id': event['id'],
                'description': event['description'],
                'text': consolidated,
                'success': success,
                'length': len(consolidated)
            })
    
    # Print summary
    print(f"\nProcessed {len(results)} events:")
    print(f"{'ID':<5} {'Success':<10} {'Length':<10} {'Description':<40}")
    print("-" * 70)
    for r in results:
        status = "✓" if r['success'] else "✗ (fallback)"
        print(f"{r['event_id']:<5} {status:<10} {r['length']:<10} {r['description'][:40]:<40}")


def example_4_extract_verses():
    """Example 4: Extract verses without consolidation."""
    print("\n" + "=" * 80)
    print("Example 4: Extract Verses Only")
    print("=" * 80)
    
    consolidator = GospelConsolidatorV2(data_dir="data")
    
    # Load events
    events = consolidator.load_chronology()
    event = events[0]  # First event
    
    print(f"\nEvent: {event['description']}")
    print(f"\nReferences:")
    for gospel in ['matthew', 'mark', 'luke', 'john']:
        ref = event.get(gospel, '').strip()
        if ref:
            print(f"  {gospel.capitalize()}: {ref}")
    
    # Extract texts
    gospel_texts = consolidator.extract_event_texts(event)
    
    print(f"\nExtracted Texts:")
    print("=" * 80)
    for gospel, text in gospel_texts.items():
        print(f"\n{gospel}:")
        print("-" * 80)
        print(text)


def example_5_batch_processing():
    """Example 5: Batch processing with progress tracking."""
    print("\n" + "=" * 80)
    print("Example 5: Batch Processing")
    print("=" * 80)
    
    consolidator = GospelConsolidatorV2(data_dir="data")
    
    # Process in batches
    batch_size = 10
    max_events = 30
    
    events = consolidator.load_chronology()[:max_events]
    
    all_results = []
    for batch_start in range(0, len(events), batch_size):
        batch_end = min(batch_start + batch_size, len(events))
        batch_events = events[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1} "
              f"(events {batch_start+1}-{batch_end})...")
        
        batch_results = []
        for i, event in enumerate(batch_events, batch_start + 1):
            consolidated, success = consolidator.consolidate_event(
                event,
                event_num=i,
                total_events=len(events),
                verbose=False
            )
            
            if consolidated:
                batch_results.append(f"{i} {consolidated}")
        
        all_results.extend(batch_results)
        print(f"  ✓ Completed {len(batch_results)} events")
    
    # Save final result
    final_narrative = " ".join(all_results)
    output_file = Path("outputs/example5_batch_output.txt")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_narrative)
    
    print(f"\n[✓] Saved {len(all_results)} events to: {output_file}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Gospel Consolidator v2 - Example Usage")
    print("=" * 80)
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_single_event()
        example_3_custom_validation()
        example_4_extract_verses()
        example_5_batch_processing()
        
        print("\n" + "=" * 80)
        print("[✓] All examples completed successfully!")
        print("=" * 80)
    
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
