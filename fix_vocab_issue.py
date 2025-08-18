#!/usr/bin/env python3

import sys
sys.path.append("src")

from pathlib import Path
import json

def fix_vocab_and_manifests():
    """Fix vocabulary and manifest files to use proper character-level vocab"""
    
    data_dir = Path("./data/gigaspeech")
    
    # Create proper character-level vocabulary
    chars = [' '] + list("'ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['<blank>']
    
    print(f"Creating character-level vocabulary with {len(chars)} characters:")
    for i, char in enumerate(chars):
        print(f"  {i:2d}: '{char}'")
    
    # Write new vocabulary
    vocab_path = data_dir / "vocab.txt"
    with open(vocab_path, 'w') as f:
        for char in chars:
            f.write(f"{char}\n")
    
    print(f"\nWrote new vocabulary to {vocab_path}")
    
    # Function to clean text
    def clean_text(text):
        """Clean text by removing punctuation tags and keeping only letters/spaces"""
        # Remove punctuation tags
        import re
        text = re.sub(r'<[^>]+>', '', text)
        # Keep only letters, spaces, and apostrophes
        text = re.sub(r"[^A-Z' ]", '', text.upper())
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Fix train manifest
    train_manifest = data_dir / "train_manifest.json"
    train_manifest_new = data_dir / "train_manifest_fixed.json"
    
    print(f"\nFixing train manifest...")
    count = 0
    with open(train_manifest, 'r') as f_in, open(train_manifest_new, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            original_text = data['text']
            cleaned_text = clean_text(original_text)
            
            if len(cleaned_text.strip()) > 0:  # Only keep non-empty texts
                data['text'] = cleaned_text
                f_out.write(json.dumps(data) + '\n')
                count += 1
                
                if count <= 5:  # Show first 5 examples
                    print(f"  Original: '{original_text}'")
                    print(f"  Cleaned:  '{cleaned_text}'")
                    print()
    
    print(f"Fixed train manifest: {count} samples")
    
    # Fix dev manifest
    dev_manifest = data_dir / "dev_manifest.json"
    dev_manifest_new = data_dir / "dev_manifest_fixed.json"
    
    print(f"\nFixing dev manifest...")
    count = 0
    with open(dev_manifest, 'r') as f_in, open(dev_manifest_new, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            original_text = data['text']
            cleaned_text = clean_text(original_text)
            
            if len(cleaned_text.strip()) > 0:  # Only keep non-empty texts
                data['text'] = cleaned_text
                f_out.write(json.dumps(data) + '\n')
                count += 1
                
                if count <= 3:  # Show first 3 examples
                    print(f"  Original: '{original_text}'")
                    print(f"  Cleaned:  '{cleaned_text}'")
                    print()
    
    print(f"Fixed dev manifest: {count} samples")
    
    # Backup originals and replace
    train_manifest.rename(data_dir / "train_manifest_original.json")
    dev_manifest.rename(data_dir / "dev_manifest_original.json")
    train_manifest_new.rename(train_manifest)
    dev_manifest_new.rename(dev_manifest)
    
    print("\nBacked up original manifests and replaced with fixed versions")
    print("Ready to restart training with proper character-level vocabulary!")

if __name__ == "__main__":
    fix_vocab_and_manifests()