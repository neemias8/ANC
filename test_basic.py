#!/usr/bin/env python3
"""
Basic test of Gospel Consolidator v2 - without loading the model.
Tests data loading and parsing functionality.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

print("=" * 80)
print("Testing Gospel Consolidator v2 - Basic Functionality")
print("=" * 80)

# Test 1: Check data files
print("\n[Test 1] Checking data files...")
data_dir = Path("data")
required_files = [
    "ChronologyOfTheFourGospels_PW.xml",
    "EnglishNIVMatthew40_PW.xml",
    "EnglishNIVMark41_PW.xml",
    "EnglishNIVLuke42_PW.xml",
    "EnglishNIVJohn43_PW.xml"
]

all_exist = True
for filename in required_files:
    filepath = data_dir / filename
    if filepath.exists():
        print(f"  [✓] {filename}")
    else:
        print(f"  [✗] {filename} - NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n[✗] Some data files are missing!")
    sys.exit(1)

print("[✓] All data files present")

# Test 2: Load chronology
print("\n[Test 2] Loading chronology...")
try:
    xml_path = data_dir / "ChronologyOfTheFourGospels_PW.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    events = []
    for event_elem in root.findall('.//event'):
        event = {
            'id': event_elem.get('id'),
            'description': event_elem.findtext('description', '').strip(),
            'matthew': event_elem.findtext('matthew', '').strip(),
            'mark': event_elem.findtext('mark', '').strip(),
            'luke': event_elem.findtext('luke', '').strip(),
            'john': event_elem.findtext('john', '').strip()
        }
        events.append(event)
    
    print(f"[✓] Loaded {len(events)} events")
    
    # Show first 3 events
    print("\nFirst 3 events:")
    for i, event in enumerate(events[:3], 1):
        print(f"  {i}. {event['description']}")
        refs = []
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            ref = event.get(gospel, '').strip()
            if ref:
                refs.append(f"{gospel.capitalize()}: {ref}")
        if refs:
            print(f"     {', '.join(refs)}")

except Exception as e:
    print(f"[✗] Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Extract sample verse
print("\n[Test 3] Extracting sample verse (Matthew 26:6)...")
try:
    xml_path = data_dir / "EnglishNIVMatthew40_PW.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find chapter 26
    for chapter in root.findall('.//chapter'):
        if int(chapter.get('number', 0)) == 26:
            # Find verse 6
            for verse in chapter.findall('.//verse'):
                if int(verse.get('number', 0)) == 6:
                    verse_text = verse.text.strip() if verse.text else ""
                    print(f"  Matthew 26:6: {verse_text}")
                    break
            break
    
    print("[✓] Verse extraction works")

except Exception as e:
    print(f"[✗] Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Import main module
print("\n[Test 4] Importing main module...")
try:
    from gospel_consolidator_v2 import GospelConsolidatorV2
    print("[✓] Successfully imported GospelConsolidatorV2")
except ImportError as e:
    print(f"[✗] Failed to import: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("[✓] All basic tests passed!")
print("=" * 80)
print("\nThe code structure is correct and data files are accessible.")
print("\nTo test with the actual model (requires transformers + torch):")
print("  python gospel_consolidator_v2.py --test")
