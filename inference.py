#!/usr/bin/env python3
"""
Complete VALLR Inference Pipeline
Video → Phonemes → Text
"""

import torch
import argparse
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from transformers import (
    VideoMAEConfig, 
    Wav2Vec2Config,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel
from Models.VALLR import VALLR
from config import get_vocab
import cv2
from decord import VideoReader, cpu
import jiwer


def load_video(video_path, num_frames=16, frame_size=(224, 224)):
    """Load and preprocess video frames."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    except Exception as e:
        print(f"Error loading video: {video_path}. Error: {e}")
        return None

    frame_count = len(vr)
    
    if frame_count < num_frames:  
        print(f"Warning: Video has only {frame_count} frames, need {num_frames}")
        return None
    
    # Sample frames uniformly
    sample_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)
    frames = []
    
    for idx in sample_indices:
        frame = vr[idx]. asnumpy()
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    # Convert to tensor:  (T, H, W, C) -> (T, C, H, W)
    video_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension:  (1, T, C, H, W)
    
    # Normalize to [0, 1]
    video_tensor = video_tensor.float() / 255.0
    
    return video_tensor


def load_part1_model(model_path, device, phoneme_vocab):
    """Load Part 1 model (Video → Phonemes)."""
    print(f"Loading Part 1 model from {model_path}...")
    
    videomae_config = VideoMAEConfig()
    wav2vec_config = Wav2Vec2Config()
    wav2vec_config.vocab_size = len(phoneme_vocab)
    
    model = VALLR(
        videomae_config=videomae_config,
        wav2vec_config=wav2vec_config,
        adapter_dim=256,
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Part 1 model loaded successfully")
    return model


def ctc_decode(logits, vocab, blank_id=0):
    """
    CTC greedy decoding with blank removal and deduplication.
    
    Args:
        logits:  (batch, time, vocab_size) - raw model outputs
        vocab: dict mapping phoneme -> index
        blank_id: index of blank token (usually 0 for <pad>)
    
    Returns:
        List of decoded phoneme sequences
    """
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    # Get most likely token at each time step
    predictions = torch.argmax(logits, dim=-1)  # (batch, time)
    
    decoded_sequences = []
    
    for pred_seq in predictions:
        # Convert to list
        pred_list = pred_seq.cpu().tolist()
        
        # CTC collapse: remove consecutive duplicates
        collapsed = []
        prev_token = None
        for token in pred_list:
            if token != prev_token:
                collapsed.append(token)
                prev_token = token
        
        # Remove blanks
        no_blanks = [t for t in collapsed if t != blank_id]
        
        # Convert indices to phonemes
        phonemes = [reverse_vocab. get(idx, '<UNK>') for idx in no_blanks]
        
        # Filter out special tokens
        phonemes = [p for p in phonemes if p not in ['<pad>', '<UNK>']]
        
        decoded_sequences.append(phonemes)
    
    return decoded_sequences


def video_to_phonemes(video_path, model, device, phoneme_vocab):
    """
    Part 1: Video → Phonemes
    
    Returns:  
        phonemes (list): List of phoneme strings
        phoneme_string (str): Space-separated phonemes
    """
    # Load video
    video_tensor = load_video(video_path, num_frames=16)
    if video_tensor is None:  
        return None, None
    
    # Run inference
    with torch.no_grad():
        video_tensor = video_tensor. to(device)
        logits, _ = model(video_tensor)  # (batch, time, vocab_size)
    
    # CTC decode
    decoded = ctc_decode(logits, phoneme_vocab, blank_id=0)
    phonemes = decoded[0]  # First (and only) sequence in batch
    
    # Join into string
    phoneme_string = " ".join(phonemes)
    
    return phonemes, phoneme_string


def load_part2_model(lora_path, device):
    """Load Part 2 model (Phonemes → Text)."""
    print(f"Loading Part 2 model from {lora_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    
    # Load base model
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    model.eval()
    
    print("Part 2 model loaded successfully")
    return model, tokenizer

def text_to_phonemes(text):
    """
    Convert text to phoneme sequence using g2p_en.  
    This simulates what Part 1 would output.
    """
    try:
        from g2p_en import G2p
    except ImportError: 
        print("Installing g2p_en...")
        import subprocess
        subprocess.check_call(["pip", "install", "g2p-en"])
        from g2p_en import G2p
    
    # Download required NLTK data
    try:
        import nltk
        nltk. data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading required NLTK data...")
        import nltk
        nltk. download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('cmudict', quiet=True)
    
    g2p = G2p()
    phonemes = g2p(text)
    
    # Clean up phonemes (remove numbers/stress markers and spaces)
    cleaned = []
    for p in phonemes: 
        p = p.strip()
        # Remove any digits (stress markers)
        p_clean = ''.join([c for c in p if not c.isdigit()])
        # Only add non-empty, non-space phonemes
        if p_clean and p_clean not in [' ', '']: 
            cleaned.append(p_clean.upper())
    
    return ' '.join(cleaned)

def test_part2_only(text, part2_model_path, device):
    """Test Part 2 model with text input (converts to phonemes first)."""
    print("="*60)
    print("Part 2 Standalone Test (Text → Phonemes → Text)")
    print("="*60)
    
    # Convert text to phonemes
    print(f"\nInput text: {text}")
    phoneme_string = text_to_phonemes(text)
    print(f"Phonemes: {phoneme_string}")
    
    # Load Part 2 model
    part2_model, tokenizer = load_part2_model(part2_model_path, device)
    
    # Generate text
    generated_text = phonemes_to_text(phoneme_string, part2_model, tokenizer, device)
    
    print(f"\nGenerated text: {generated_text}")
    print(f"Original text: {text}")
    print("="*60)
    
    return generated_text

def phonemes_to_text(phoneme_string, model, tokenizer, device, max_new_tokens=100):
    """
    Part 2: Phonemes → Text
    
    Args:
        phoneme_string (str): Space-separated phonemes (e.g., "DH AH K AE T")
        model: Part 2 model
        tokenizer:  Tokenizer
        device: torch device
        max_new_tokens (int): Max tokens to generate
    
    Returns:  
        generated_text (str): The predicted text
    """
    # Format prompt (matching training format)
    prompt = (
        "<S2S>\n"
        "<PHONEMES>\n"
        f"{phoneme_string}\n"
        "</PHONEMES>\n"
        "<TEXT>\n"
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer. eos_token_id
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the generated text (after <TEXT>)
    if "<TEXT>" in full_output:  
        generated_text = full_output.split("<TEXT>")[-1].strip()
        # Remove any trailing special tokens
        for tag in ["</S2S>", tokenizer.eos_token]:  
            if generated_text. endswith(tag):
                generated_text = generated_text[:-len(tag)].strip()
    else:
        generated_text = full_output
    
    return generated_text


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis.
    
    Args:
        reference (str): Ground truth text
        hypothesis (str): Predicted text
    
    Returns:
        float: WER as percentage
    """
    if not reference or not hypothesis:
        return None
    
    try:
        # Normalize:  uppercase and strip
        reference = reference.upper().strip()
        hypothesis = hypothesis.upper().strip()
        
        # Calculate WER
        wer = jiwer.wer(reference, hypothesis)
        return wer * 100  # Convert to percentage
    except: 
        return None


def find_videos_in_folder(folder_path):
    """Find all video files in folder and subfolders."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    video_files = []
    
    folder = Path(folder_path)
    for ext in video_extensions:
        video_files.extend(folder.rglob(f"*{ext}"))
    
    return [str(v) for v in video_files]


def load_ground_truth(video_path):
    """Load ground truth text from .txt file if it exists."""
    txt_path = video_path.rsplit('.', 1)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            content = f.read().strip()
            # Extract text after "Text:" if present
            if "Text:" in content:
                return content.split("Text:")[-1].strip()
            return content
    return None


def process_folder(folder_path, part1_model_path, part2_model_path, output_file, part1_only, device):
    """Process all videos in a folder."""
    
    # Find all videos
    video_files = find_videos_in_folder(folder_path)
    
    if not video_files:
        print(f"No videos found in {folder_path}")
        return
    
    print(f"Found {len(video_files)} videos")
    
    # Load models
    phoneme_vocab = get_vocab()
    part1_model = load_part1_model(part1_model_path, device, phoneme_vocab)
    
    part2_model = None
    tokenizer = None
    if not part1_only:
        part2_model, tokenizer = load_part2_model(part2_model_path, device)
    
    # Process videos
    results = []
    total_wer = 0
    wer_count = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Part 1: Video → Phonemes
            phonemes, phoneme_string = video_to_phonemes(
                video_path, 
                part1_model, 
                device, 
                phoneme_vocab
            )
            
            if phoneme_string is None:
                results.append({
                    "video": video_path,
                    "error": "Failed to load video"
                })
                continue
            
            result = {
                "video": video_path,
                "phonemes": phoneme_string,
            }
            
            # Load ground truth if available
            ground_truth = load_ground_truth(video_path)
            if ground_truth:
                result["ground_truth"] = ground_truth
            
            # Part 2: Phonemes → Text (if not part1_only)
            if not part1_only and phoneme_string:
                text = phonemes_to_text(
                    phoneme_string,
                    part2_model,
                    tokenizer,
                    device
                )
                result["predicted_text"] = text
                
                # Calculate WER if ground truth exists
                if ground_truth: 
                    wer = calculate_wer(ground_truth, text)
                    if wer is not None:
                        result["wer"] = round(wer, 2)
                        total_wer += wer
                        wer_count += 1
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "video": video_path,
                "error": str(e)
            })
    
    # Calculate average WER
    avg_wer = total_wer / wer_count if wer_count > 0 else None
    
    # Prepare output with summary
    output_data = {
        "summary": {
            "total_videos": len(results),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
        },
        "results": results
    }
    
    if avg_wer is not None: 
        output_data["summary"]["average_wer"] = round(avg_wer, 2)
        output_data["summary"]["wer_samples"] = wer_count
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Total videos:  {output_data['summary']['total_videos']}")
    print(f"Successfully processed: {output_data['summary']['successful']}")
    print(f"Failed: {output_data['summary']['failed']}")
    
    if avg_wer is not None: 
        print(f"\nWord Error Rate (WER):")
        print(f"  Average WER: {avg_wer:. 2f}%")
        print(f"  Samples with ground truth: {wer_count}")
    
    print(f"\nResults saved to {output_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="VALLR Inference Pipeline")
    parser.add_argument("--video", type=str, help="Path to single input video")
    parser.add_argument("--folder", type=str, help="Path to folder containing videos")
    parser.add_argument("--test-part2", type=str, help="Test Part 2 only with text input (e.g., 'hello world')")
    parser.add_argument("--part1-model", type=str, help="Path to Part 1 model checkpoint (. pth)")
    parser.add_argument("--part2-model", type=str, help="Path to Part 2 LoRA model directory")
    parser.add_argument("--output", type=str, default="results. json",
                       help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device:  'cuda' or 'cpu'")
    parser.add_argument("--part1-only", action="store_true",
                       help="Run Part 1 only (Video → Phonemes)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Max tokens to generate in Part 2")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Mode 1: Test Part 2 only
    if args.test_part2:
        if not args.part2_model:
            parser.error("--part2-model is required for --test-part2")
        test_part2_only(args. test_part2, args.part2_model, device)
        return
    
    # Validate arguments for video/folder modes
    if not args.video and not args.folder:
        parser.error("Must provide either --video, --folder, or --test-part2")
    
    if args.video and args.folder:
        parser.error("Provide only one of --video or --folder")
    
    if not args.part1_model:
        parser.error("--part1-model is required for video/folder processing")
    
    if not args.part2_model and not args.part1_only:
        parser.error("--part2-model is required unless --part1-only is specified")
    
    # Mode 2: Process entire folder
    if args.folder:
        process_folder(
            args.folder,
            args.part1_model,
            args.part2_model,
            args.output,
            args.part1_only,
            device
        )
    # Mode 3: Process single video
    else:
        phoneme_vocab = get_vocab()
        part1_model = load_part1_model(args. part1_model, device, phoneme_vocab)
        
        phonemes, phoneme_string = video_to_phonemes(
            args.video,
            part1_model,
            device,
            phoneme_vocab
        )
        
        print(f"\nPredicted phonemes:  {phoneme_string}")
        
        if not args.part1_only: 
            if not phoneme_string:
                print("Warning: No phonemes predicted.  Skipping Part 2...")
            else:
                part2_model, tokenizer = load_part2_model(args.part2_model, device)
                text = phonemes_to_text(phoneme_string, part2_model, tokenizer, device, args. max_tokens)
                print(f"Predicted text: {text}")
                
                # Load ground truth if available
                ground_truth = load_ground_truth(args.video)
                if ground_truth: 
                    print(f"Ground truth: {ground_truth}")
                    
                    # Calculate WER
                    wer = calculate_wer(ground_truth, text)
                    if wer is not None:
                        print(f"WER: {wer:. 2f}%")


if __name__ == "__main__": 
    main()