#!/usr/bin/env python3
"""
Quick diagnostic for Part 1 model to understand why it's predicting only blanks.
"""

import torch
import numpy as np
from Models.VALLR import VALLR
from transformers import VideoMAEConfig, Wav2Vec2Config
from config import get_vocab

def analyze_model(model_path, device='cuda'):
    """Analyze the model's behavior in detail."""
    
    # Load vocab
    vocab = get_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Blank token (<pad>): index {vocab['<pad>']}")
    
    # Initialize model
    videomae_config = VideoMAEConfig()
    wav2vec_config = Wav2Vec2Config()
    wav2vec_config.vocab_size = len(vocab)
    
    model = VALLR(
        videomae_config=videomae_config,
        wav2vec_config=wav2vec_config,
        adapter_dim=256,
    )
    
    # Load weights
    print(f"\nLoading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create dummy input (16 frames, 3 channels, 224x224)
    dummy_video = torch.randn(1, 16, 3, 224, 224).to(device)
    
    print("\n" + "="*60)
    print("TESTING MODEL WITH RANDOM INPUT")
    print("="*60)
    
    with torch.no_grad():
        logits, feats = model(dummy_video)
        
        print(f"\nLogits shape: {logits.shape}")
        print(f"Logits range: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
        
        # Check predictions
        predictions = torch.argmax(logits, dim=-1)
        pred_list = predictions[0].cpu().numpy()
        
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"First 50 predictions: {pred_list[:50].tolist()}")
        print(f"Unique predicted indices: {np.unique(pred_list).tolist()}")
        
        # Check if blank dominates
        blank_id = vocab['<pad>']
        blank_count = (predictions == blank_id).sum().item()
        total_count = predictions.numel()
        blank_pct = 100 * blank_count / total_count
        
        print(f"\n" + "="*60)
        print(f"BLANK TOKEN ANALYSIS:")
        print(f"="*60)
        print(f"Blank predictions: {blank_count}/{total_count} ({blank_pct:.1f}%)")
        
        # Analyze logits for blank vs non-blank
        blank_logits = logits[:, :, blank_id].mean().item()
        non_blank_logits = logits[:, :, 1:].mean().item()
        
        print(f"\nAverage logits:")
        print(f"  Blank token (index {blank_id}): {blank_logits:.4f}")
        print(f"  Non-blank tokens (avg): {non_blank_logits:.4f}")
        print(f"  Difference: {blank_logits - non_blank_logits:.4f}")
        
        if blank_pct > 90:
            print("\n⚠️  WARNING: Model is predicting >90% blanks!")
            print("   This suggests the model did NOT learn properly.")
            print("   Possible issues:")
            print("   1. Learning rate was too low during training")
            print("   2. Model didn't train long enough")
            print("   3. Training data had issues")
        elif blank_pct > 50:
            print("\n⚠️  WARNING: Model is predicting >50% blanks!")
            print("   This is higher than expected for normal speech.")
        else:
            print("\n✓ Blank percentage looks reasonable!")
        
        # Check logits distribution
        print(f"\n" + "="*60)
        print(f"LOGITS DISTRIBUTION:")
        print(f"="*60)
        for i in range(min(10, logits.shape[-1])):
            mean_logit = logits[:, :, i].mean().item()
            max_logit = logits[:, :, i].max().item()
            print(f"  Token {i:2d}: mean={mean_logit:6.3f}, max={max_logit:6.3f}")
        
        print("\n" + "="*60)
        print("DIAGNOSIS:")
        print("="*60)
        
        if blank_logits > non_blank_logits + 2.0:
            print("❌ PROBLEM: Blank logits are much higher than non-blank logits")
            print("   This model was likely trained with too low learning rate.")
            print("   The model collapsed to always predicting blank.")
            print("\n   SOLUTION: Retrain with higher learning rate (5e-5 → 3e-4)")
        elif len(np.unique(pred_list)) < 5:
            print("❌ PROBLEM: Model only predicts very few different tokens")
            print("   The model didn't learn the phoneme diversity.")
            print("\n   SOLUTION: Retrain with higher learning rate")
        else:
            print("✓ Model seems to be working reasonably!")
            print("  It's predicting diverse tokens, not just blanks.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "./checkpoints/vallr_part2.pth"
    
    print(f"Analyzing model: {model_path}\n")
    analyze_model(model_path)
