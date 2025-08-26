#!/usr/bin/env python3
"""
debug_dataset.py - Debug script to understand dataset structure
Run this to see what's actually in your dataset files
"""

import json
from pathlib import Path
import sys

def inspect_file(file_path):
    """Inspect a dataset file and show its structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path}")
    print('='*60)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    # Try to load the file
    if file_path.endswith('.jsonl'):
        print("Format: JSONL (line-delimited JSON)")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Total lines: {len(lines)}")
            
            if lines:
                # Check first few lines
                for i, line in enumerate(lines[:3]):
                    try:
                        item = json.loads(line)
                        print(f"\nLine {i+1} structure:")
                        print(f"  Keys: {list(item.keys())}")
                        
                        # Show sample content
                        for key in item.keys():
                            value = item[key]
                            if isinstance(value, str):
                                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
                            else:
                                print(f"  {key}: {type(value).__name__}")
                                
                        # Check for formal/informal patterns
                        if 'text' in item:
                            text = item['text']
                            if 'informal statement' in text and 'formal statement' in text:
                                print("  ✓ Contains 'informal statement' and 'formal statement' markers in 'text' field")
                                # Try to parse
                                if '.formal statement' in text:
                                    parts = text.split('.formal statement')
                                    informal_part = parts[0].replace('informal statement', '').strip()
                                    formal_part = parts[1].strip() if len(parts) > 1 else ''
                                    print(f"  Parsed informal: {informal_part[:50]}...")
                                    print(f"  Parsed formal: {formal_part[:50]}...")
                                    
                    except json.JSONDecodeError as e:
                        print(f"  ❌ Failed to parse line {i+1}: {e}")
                        
                return "jsonl"
                
    elif file_path.endswith('.json'):
        print("Format: JSON")
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                if isinstance(data, list):
                    print(f"Structure: JSON array with {len(data)} items")
                    
                    if data:
                        # Check first few items
                        for i, item in enumerate(data[:3]):
                            print(f"\nItem {i+1} structure:")
                            if isinstance(item, dict):
                                print(f"  Keys: {list(item.keys())}")
                                
                                # Show sample content
                                for key in item.keys():
                                    value = item[key]
                                    if isinstance(value, str):
                                        print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
                                    else:
                                        print(f"  {key}: {type(value).__name__}")
                                        
                                # Check for formal/informal
                                if 'formal' in item and 'informal' in item:
                                    print("  ✓ Has 'formal' and 'informal' keys")
                                elif 'text' in item:
                                    text = item['text']
                                    if 'informal statement' in text and 'formal statement' in text:
                                        print("  ✓ Has 'text' field with formal/informal markers")
                            else:
                                print(f"  Type: {type(item).__name__}")
                                
                elif isinstance(data, dict):
                    print(f"Structure: JSON object with {len(data)} keys")
                    print(f"Keys (first 10): {list(data.keys())[:10]}")
                    
                    # Check if it has a data field
                    for key in ['data', 'examples', 'samples', 'items']:
                        if key in data:
                            print(f"\nFound '{key}' field with {len(data[key])} items")
                            if isinstance(data[key], list) and data[key]:
                                first_item = data[key][0]
                                if isinstance(first_item, dict):
                                    print(f"  First item keys: {list(first_item.keys())}")
                                    
                    # Check first few entries
                    for i, (key, value) in enumerate(list(data.items())[:3]):
                        print(f"\nEntry '{key}':")
                        if isinstance(value, dict):
                            print(f"  Keys: {list(value.keys())}")
                            if 'formal' in value and 'informal' in value:
                                print("  ✓ Has 'formal' and 'informal' keys")
                            elif 'text' in value:
                                print("  Has 'text' field")
                else:
                    print(f"Unexpected structure: {type(data).__name__}")
                    
                return "json"
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    else:
        print(f"Unknown format: {file_path}")
        return None

def main():
    """Main inspection function"""
    print("\n" + "="*80)
    print("DATASET STRUCTURE INSPECTOR")
    print("="*80)
    
    # Define paths to check
    data_dir = Path("./math_alignment_dataset")
    
    files_to_check = [
        data_dir / "train.jsonl",
        data_dir / "train.json",
        data_dir / "statements_part1.json",
        data_dir / "val.jsonl",
        data_dir / "val.json",
        data_dir / "valid_clean.json",
        data_dir / "test.jsonl",
        data_dir / "test.json",
        data_dir / "test_clean.json",
    ]
    
    found_files = {}
    
    for file_path in files_to_check:
        if file_path.exists():
            file_type = inspect_file(str(file_path))
            if file_type:
                found_files[file_path.name] = file_type
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if found_files:
        print("\nFound files:")
        for name, ftype in found_files.items():
            print(f"  • {name} ({ftype})")
            
        print("\n📝 Recommendations:")
        print("1. Ensure all files have 'formal' and 'informal' keys in each item")
        print("2. Or have a 'text' field with pattern: 'informal statement [...].formal statement [...]'")
        print("3. Check that the data is not empty")
        print("4. Verify JSON syntax is correct")
    else:
        print("\n❌ No dataset files found in ./math_alignment_dataset/")
        print("\nPlease ensure you have run the dataset curation script first.")
    
    # Try to load one file properly to show a working example
    print("\n" + "="*80)
    print("ATTEMPTING TO LOAD DATA")
    print("="*80)
    
    for file_path in files_to_check:
        if file_path.exists():
            print(f"\nTrying to load: {file_path}")
            data_items = []
            
            try:
                if str(file_path).endswith('.jsonl'):
                    with open(file_path, 'r') as f:
                        for line in f:
                            item = json.loads(line)
                            # Try different extraction methods
                            if 'formal' in item and 'informal' in item:
                                data_items.append(item)
                            elif 'text' in item:
                                text = item['text']
                                if 'informal statement' in text and '.formal statement' in text:
                                    parts = text.split('.formal statement')
                                    if len(parts) >= 2:
                                        informal = parts[0].replace('informal statement', '').strip()
                                        formal = parts[1].strip()
                                        data_items.append({'formal': formal, 'informal': informal})
                                        
                elif str(file_path).endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if 'formal' in item and 'informal' in item:
                                        data_items.append(item)
                                    elif 'text' in item:
                                        text = item['text']
                                        if 'informal statement' in text and '.formal statement' in text:
                                            parts = text.split('.formal statement')
                                            if len(parts) >= 2:
                                                informal = parts[0].replace('informal statement', '').strip()
                                                formal = parts[1].strip()
                                                data_items.append({'formal': formal, 'informal': informal})
                
                if data_items:
                    print(f"✓ Successfully loaded {len(data_items)} items")
                    print(f"Sample item:")
                    print(f"  Formal: {data_items[0]['formal'][:100]}...")
                    print(f"  Informal: {data_items[0]['informal'][:100]}...")
                    break
                else:
                    print(f"⚠ File exists but no valid items found")
                    
            except Exception as e:
                print(f"❌ Error loading: {e}")

if __name__ == "__main__":
    main()