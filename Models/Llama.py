import re
import torch
import os
import sys
import csv
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

import pronouncing  # CMUdict-based

# Add parent directory to path for Data module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------- Phoneme utilities (ARPAbet via pronouncing) ----------

_WORD_RE = re.compile(r"[A-Za-z']+")

def strip_stress(arpabet: str) -> str:
    # Remove 0/1/2 stress digits (e.g., AH0 -> AH)
    return re.sub(r"\d", "", arpabet)

def text_to_arpabet_words(text: str) -> List[str]:
    """
    Converts text into a list of word-level ARPAbet strings (stress-stripped).
    Falls back to the raw word if not in CMUdict to avoid losing alignment.
    """
    words = _WORD_RE.findall(text)
    arpawords = []
    for w in words:
        lw = w.lower()
        phones = pronouncing.phones_for_word(lw)
        if phones:
            # choose first pronunciation; strip stress
            arp = strip_stress(phones[0])
            arpawords.append(arp)
        else:
            # Fallback: mark as unknown token; you can also just keep the grapheme
            # arpawords.append(lw)  # alternative
            arpawords.append(f"UNK({lw})")
    return arpawords

def text_to_phoneme_line(text: str) -> str:
    """
    Formats phonemes as word-separated with ' | ' between words.
    Example: "the cat" -> "DH AH | K AE T"
    """
    arpawords = text_to_arpabet_words(text)
    return " ".join(arpawords)


# ---------- Dataset building (phonemes -> text) ----------

TAGS = ["<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>"]

def clean_text(text: str) -> str:
    """
    Clean Wikipedia markup and convert to LRS3 style:
    - All uppercase
    - Only letters, spaces, and apostrophes
    - No punctuation, numbers, or special characters
    """
    if not text:
        return ""
    
    # Remove URLs and email patterns
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'\S+\.(com|net|org|edu|gov|io|co)\S*', '', text)
    
    # Remove Wikipedia section headers (= = = Title = = =)
    text = re.sub(r'=+\s*([^=]+)\s*=+', r'\1', text)
    
    # Remove ALL brackets and parentheses with content
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^\}]*\}', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove time stamps and numbers with colons
    text = re.sub(r'\d{1,2}\s*:\s*\d{2}', '', text)
    
    # Remove standalone years and all numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove markdown/wiki formatting
    text = re.sub(r'[*#_]{2,}([^*#_]+)[*#_]{2,}', r'\1', text)
    text = re.sub(r'[*#_]([^*#_]+)[*#_]', r'\1', text)
    text = re.sub(r'[*#_]+', '', text)
    
    # Replace @-@ with spaces (not hyphens)
    text = re.sub(r'@\s*[-–—]\s*@', ' ', text)
    text = re.sub(r'@', '', text)
    
    # Remove ALL punctuation except apostrophes (keep contractions like don't, I'm)
    # Remove: . , ! ? ; : - — – _ / \ | ~ ` " etc.
    text = re.sub(r'[^\w\s\']', ' ', text)  # Keep only word chars, spaces, apostrophes
    
    # Remove standalone punctuation
    text = re.sub(r'\s+[^\w\s]\s+', ' ', text)
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to UPPERCASE (LRS3 style)
    text = text.upper()
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Final checks
    if len(text) < 3:
        return ""
    
    # Skip if still has weird characters (after uppercasing, only A-Z, space, apostrophe allowed)
    if not all(c.isalpha() or c.isspace() or c == "'" for c in text):
        return ""
    
    return text


def build_example(text: str) -> Dict[str, str]:
    """
    Builds one training pair where the INPUT is phonemes and the TARGET is original text.
    We’ll create:
      - prompt: everything up to (and including) the <TEXT> line
      - target: the original text (model should generate this)
    """
    text = (text or "").strip()
    if not text:
        return None
    
    # Clean Wikipedia markup
    text = clean_text(text)
    if not text or len(text) < 5:
        return None
    
    # Trim to reasonable length instead of skipping
    # LRS3 is 4-23 words, so let's target 5-25 words
    words = text.split()
    if len(words) < 3:  # Too short, skip
        return None
    if len(words) > 25:  # Too long, trim to first 25 words
        words = words[:25]
        text = " ".join(words)

    phon_line = text_to_phoneme_line(text)
    if not phon_line:
        return None

    prompt = (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phon_line}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )
    target = text  # what we want the model to generate
    full = prompt + target
    return {"prompt": prompt, "target": target, "full": full}

def prepare_split(split: str):
    base = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    base = base.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)

    def mapper(batch):
        outs = [build_example(t) for t in batch["text"]]
        # filter Nones (empty lines etc.)
        outs = [o for o in outs if o is not None]
        if not outs:
            return {"prompt": [], "target": [], "full": []}
        return {
            "prompt": [o["prompt"] for o in outs],
            "target": [o["target"] for o in outs],
            "full":   [o["full"]   for o in outs],
        }

    ds = base.map(mapper, batched=True, remove_columns=base.column_names)
    return ds


# ---------- Tokenization with prefix-masked labels ----------

def make_tokenize_fn(tokenizer, max_length: int = 512, min_target_tokens: int = 4):
    """
    Ensures every example has at least `min_target_tokens` supervised tokens.
    We encode prompt and target separately, then truncate the prompt to leave room.
    """
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "pad_token_id must be set"

    def _tok(batch):
        input_ids_batch, attn_batch, labels_batch = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            # Encode WITHOUT adding extra special tokens for prompt
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            # Encode target WITH eos token so model learns to stop
            t = tokenizer(target, add_special_tokens=False)["input_ids"]
            # Add EOS token after target
            t = t + [tokenizer.eos_token_id]

            # Skip pathological cases
            if len(t) == 0:
                continue

            # Reserve space for target
            max_prompt_len = max_length - min_target_tokens
            if max_prompt_len <= 0:
                continue

            # Truncate prompt to leave room
            if len(p) > max_prompt_len:
                p = p[:max_prompt_len]

            # Fit as much target as possible, but keep at least min_target_tokens
            space = max_length - len(p)
            if space < min_target_tokens:
                # Even after truncating prompt, no room left → skip example
                continue
            t = t[:space]

            ids = p + t
            attn = [1] * len(ids)
            labs = ([-100] * len(p)) + t[:]  # supervise only target

            # Pad to max_length
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids  += [pad_id] * pad_len
                attn += [0] * pad_len
                labs += [-100] * pad_len

            input_ids_batch.append(ids)
            attn_batch.append(attn)
            labels_batch.append(labs)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels": labels_batch,
        }
    return _tok

def debug_supervision(ds, name):
    import numpy as np
    import random
    n = min(2000, len(ds))
    cnt = 0
    for i in range(n):
        labs = ds[i]["labels"]
        if any(l != -100 for l in labs):
            cnt += 1
    print(f"[{name}] examples with at least 1 supervised token: {cnt}/{n}")

# ---------- Custom collator (keep labels, just pad if needed) ----------

# ---------- Custom collator (keep labels, just pad if needed) ----------

class MetricsLoggerCallback(TrainerCallback):
    """
    Custom callback to log training metrics to a CSV file.
    Logs: epoch, step, train_loss, eval_loss, learning_rate
    """
    def __init__(self, log_file):
        self.log_file = log_file
        self.log_dir = Path(log_file).parent
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'train_loss', 'eval_loss', 'learning_rate', 'timestamp'])
        
        print(f"📊 Logging metrics to: {self.log_file}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens (every logging_steps)."""
        if logs is None:
            return
        
        # Get current values
        epoch = logs.get('epoch', state.epoch)
        step = state.global_step
        train_loss = logs.get('loss', '')
        eval_loss = logs.get('eval_loss', '')
        lr = logs.get('learning_rate', '')
        
        # Get timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, train_loss, eval_loss, lr, timestamp])

class CausalLMDataCollator(DataCollatorWithPadding):
    """
    Uses tokenizer padding for inputs. Expects 'labels' already provided;
    pads labels with -100 to match input length.
    """
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels")
        batch = super().__call__(features)

        max_len = batch["input_ids"].shape[1]
        padded = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded.append(lab)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


# ---------- Main training with LoRA ----------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LLaMA for Phoneme-to-Text")
    parser.add_argument("--test", action="store_true", help="Run in test mode with small subset")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "lrs3"],
                       help="Dataset to use: 'wikitext' (default) or 'lrs3'")
    parser.add_argument("--lrs3-text-dir", type=str, 
                       default="/home/rishabhjain/Desktop/Datasets/lrs3_rf/lrs3/lrs3_text_seg16s",
                       help="Path to LRS3 text directory (only used if --dataset=lrs3)")
    parser.add_argument("--train-samples", type=int, default=None, help="Number of training samples (default: all)")
    parser.add_argument("--val-samples", type=int, default=None, help="Number of validation samples (default: all)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (default: 1)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/llama_phonemes_to_text_lora", help="Output directory")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    
    args = parser.parse_args()
    
    # 1) Load dataset
    if args.dataset == "lrs3":
        print("Loading LRS3 dataset (trainval + pretrain, excluding test)...")
        from Data.Llama_lrs3_dataset import prepare_lrs3_dataset
        
        max_samples = 100 if args.test else None
        train_ds, val_ds = prepare_lrs3_dataset(
            args.lrs3_text_dir,
            max_samples=max_samples,
            exclude_test=True  # Never include test in training
        )
    else:  # wikitext
        print("Loading WikiText-2 dataset...")
        train_ds = prepare_split("train")
        val_ds = prepare_split("validation")
    
    # Apply test mode or custom sample sizes
    if args.test:
        print("🧪 TEST MODE: Using small subset")
        train_samples = 100
        val_samples = 50
        epochs = 2
        output_dir = "./checkpoints/part2/test/adapter"
    else:
        train_samples = args.train_samples
        val_samples = args.val_samples
        epochs = args. epochs
        output_dir = args.output_dir
    
    # Subset datasets if specified
    if train_samples is not None:
        train_ds = train_ds.select(range(min(train_samples, len(train_ds))))
        print(f"Using {len(train_ds)} training samples")
    
    if val_samples is not None:
        val_ds = val_ds.select(range(min(val_samples, len(val_ds))))
        print(f"Using {len(val_ds)} validation samples")
    
    print(f"Dataset sizes: {len(train_ds)} train, {len(val_ds)} val")

    # 2) Tokenizer
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add tags and ensure a pad token
    special = {"additional_special_tokens":  TAGS}
    added = tokenizer.add_special_tokens(special)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA convention

    # 3) Tokenize with prompt-masked labels
    print("Tokenizing dataset with masked labels...")
    tok_fn = make_tokenize_fn(tokenizer, max_length=512)
    tokenized_train = train_ds. map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val   = val_ds.map(tok_fn,   batched=True, remove_columns=val_ds.column_names)

    # 4) Load base model, resize embeddings, then wrap with LoRA
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    if added > 0:
        model. resize_token_embeddings(len(tokenizer))

    # --- LoRA config (typical for LLaMA) ---
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,  # Increased from 0.05 to prevent overfitting
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # attention
            "gate_proj", "up_proj", "down_proj"          # MLP
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check

    # 5) Collator & args
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir if not args.test else "./checkpoints/Llama_lora_training",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.05,  # Increased from 0.01 to prevent overfitting
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=3,  # Keep more checkpoints to select best
        load_best_model_at_end=True,  # Load best checkpoint at end
        metric_for_best_model="eval_loss",  # Use eval_loss to select best
        fp16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        report_to="none",
    )

    vocab = model.get_input_embeddings().num_embeddings
    print("tokenizer/model vocab:", len(tokenizer), vocab)
    debug_supervision(tokenized_train, "train")
    debug_supervision(tokenized_val, "val")

    # Create metrics logger callback
    log_file = Path(args.output_dir) / "training_log.csv"
    metrics_logger = MetricsLoggerCallback(log_file)

    # 6) Train
    print("Setting up Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[metrics_logger],
    )

    print("Starting training...")
    trainer.train()

    # 7) Save adapters (default) + tokenizer
    save_path = args.output_dir if not args.test else "./checkpoints/llama_phonemes_to_text_lora_test"
    print(f"Saving the LoRA adapters to {save_path}...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # --- Optional: export a merged full model (fp16, larger) ---
    if not args.test:  # Skip merging in test mode
        print("Merging LoRA into base weights for export...")
        merged = model.merge_and_unload()
        merged_path = save_path + "_merged"
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

    # 8) Evaluate
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__": 
    main()