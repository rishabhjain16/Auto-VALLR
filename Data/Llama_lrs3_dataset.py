"""
LRS3 Dataset for LLaMA Part 2 Training (Phoneme → Text)
Uses LRS3 transcripts as clean training data instead of WikiText
"""

from datasets import Dataset
from pathlib import Path
import pronouncing
from typing import Dict, List
import re


TAGS = [
    "<PHONEMES>",
    "<PHONEME_SEQUENCE>",
    "</PHONEME_SEQUENCE>",
    "<TEXT>"
]


def text_to_phoneme_line(text: str) -> str:
    """Convert text to space-separated phonemes using pronouncing library"""
    if not text:
        return ""
    
    # Remove digits and special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s\']', '', text)
    words = text.lower().split()
    
    phoneme_list = []
    for word in words:
        phones = pronouncing.phones_for_word(word)
        if phones:
            # Take first pronunciation, remove stress markers
            phone_seq = phones[0]
            phone_seq = re.sub(r'[0-9]', '', phone_seq)  # Remove stress
            phoneme_list.append(phone_seq)
    
    return " ".join(phoneme_list)


def build_example(text: str) -> Dict[str, str]:
    """
    Build training example from LRS3 text.
    Returns dict with prompt, target, and full text.
    """
    text = (text or "").strip()
    if not text or len(text) < 5:
        return None
    
    # LRS3 text is already clean (UPPERCASE, letters/spaces/apostrophes)
    # Just validate it's reasonable length (4-30 words for LRS3 range)
    words = text.split()
    if len(words) < 3:  # Too short
        return None
    if len(words) > 30:  # Trim long sequences
        words = words[:30]
        text = " ".join(words)
    
    # Convert to phonemes
    phon_line = text_to_phoneme_line(text)
    if not phon_line:
        return None
    
    # Build prompt/target in same format as WikiText version
    prompt = (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phon_line}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )
    target = text  # LRS3 text is already clean
    full = prompt + target
    
    return {"prompt": prompt, "target": target, "full": full}


def load_lrs3_texts(lrs3_text_dir: str, exclude_test: bool = True) -> List[str]:
    """
    Load all text files from LRS3 text directory.
    
    Args:
        lrs3_text_dir: Path to lrs3_text_seg16s directory
        exclude_test: If True, excludes test split (default: True)
    
    Returns:
        List of text strings (one per file)
    """
    text_root = Path(lrs3_text_dir)
    all_texts = []
    
    # Get all available splits in the directory
    available_splits = [d.name for d in text_root.iterdir() if d.is_dir()]
    
    # Filter out test if requested
    if exclude_test and "test" in available_splits:
        available_splits.remove("test")
        print(f"ℹ️  Excluding 'test' split from training data")
    
    print(f"Loading from splits: {', '.join(available_splits)}")
    
    for split in sorted(available_splits):
        split_dir = text_root / split
        
        # Get all .txt files recursively
        text_files = sorted(split_dir.glob("**/*.txt"))
        
        print(f"Loading {len(text_files)} text files from {split}...")
        count_before = len(all_texts)
        
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text and len(text) > 5:
                        all_texts.append(text)
            except Exception as e:
                # Skip corrupted files
                continue
        
        added = len(all_texts) - count_before
        print(f"✅ Added {added} texts from {split} (total: {len(all_texts)})")
    
    return all_texts


def prepare_lrs3_dataset(lrs3_text_dir: str, 
                         train_val_split: float = 0.95,
                         max_samples: int = None,
                         exclude_test: bool = True) -> tuple:
    """
    Prepare train/val datasets from LRS3 text files.
    Uses ALL available data (trainval + pretrain) except test set.
    
    Args:
        lrs3_text_dir: Path to lrs3_text_seg16s directory
        train_val_split: Fraction for training (default: 0.95 = 95% train, 5% val)
        max_samples: Limit number of samples for testing (default: None = all)
        exclude_test: Exclude test split from training (default: True)
    
    Returns:
        (train_dataset, val_dataset) as HuggingFace Dataset objects
    """
    print(f"\n{'='*60}")
    print(f"Loading LRS3 Dataset for LLaMA Part 2 Training")
    print(f"{'='*60}\n")
    
    # Load all texts (automatically uses all splits except test)
    all_texts = load_lrs3_texts(lrs3_text_dir, exclude_test=exclude_test)
    
    if max_samples:
        all_texts = all_texts[:max_samples]
        print(f"⚠️  Limited to {max_samples} samples for testing")
    
    # Build examples
    print(f"\nConverting {len(all_texts)} texts to phoneme→text pairs...")
    examples = []
    skipped = 0
    
    for text in all_texts:
        example = build_example(text)
        if example:
            examples.append(example)
        else:
            skipped += 1
    
    print(f"✅ Created {len(examples)} valid examples ({skipped} skipped)")
    
    # Split into train/val
    split_idx = int(len(examples) * train_val_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(train_examples)} examples")
    print(f"   Validation: {len(val_examples)} examples")
    
    # Convert to HuggingFace datasets
    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)
    
    print(f"\n{'='*60}\n")
    
    return train_ds, val_ds


if __name__ == "__main__":
    # Test the dataset loader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrs3-text-dir", type=str, 
                       default="/home/rishabhjain/Desktop/Datasets/lrs3_rf/lrs3/lrs3_text_seg16s",
                       help="Path to LRS3 text directory")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Max samples for testing")
    parser.add_argument("--include-test", action="store_true",
                       help="Include test split (default: exclude test)")
    args = parser.parse_args()
    
    # Test loading
    train_ds, val_ds = prepare_lrs3_dataset(
        args.lrs3_text_dir,
        max_samples=args.max_samples,
        exclude_test=not args.include_test
    )
    
    # Show example
    print("\n📄 Example training sample:")
    print("-" * 60)
    example = train_ds[0]
    print(f"PROMPT:\n{example['prompt']}")
    print(f"TARGET: {example['target']}")
    print("-" * 60)
