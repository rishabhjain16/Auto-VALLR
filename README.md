# VALLR Training Guide with Auto_AVSR Dataset

This guide provides step-by-step instructions for training VALLR on the Auto_AVSR (LRS3) dataset for continuous speech lip-reading. 
**This repo was foked from the VALLR repo and is currently under development.**

Note: The VALLR paper is quite interesting, however, I found many missing components in their Github codebase, so I decided to improve on it and make my own version of it as per my usecase. I also use Auto-AVSR based data cleaning to work with full sentences rather than the word-level segmentation they do. At least, that's the plan.

Update: I tried a few training, however its hard to get CTC alignment to converge at sentence-level. In the original Vall-R Codebase, authors use word-level predictions with CTC which seemsed to have worked for them as per the reported results. However, in this codebase, I am finding it difficult to converge it for sentence-level. I have not tried to make it work at work-level since I am adapting the Auto-AVSR data processing pipeline to work with my datasets. I am open to ideas. 
---

## Table of Contents
- [Setup](#setup)
- [Overview](#overview)
- [Architecture Modifications](#architecture-modifications)
- [Dataset Preparation](#dataset-preparation)
- [Training Strategy](#training-strategy)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/rishabhjain16/Auto-VALLR.git
cd Auto-VALLR
```

### 2. Create Conda Environment
```bash
# Create a new conda environment with Python 3.9 or 3.10
conda create -n vallr python=3.10 -y
conda activate vallr
```

### 3. Install Dependencies
```bash
# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other requirements
pip install -r requirements.txt
```

### 4. Download Pre-trained Models (if needed)
Place any pre-trained checkpoints in the `checkpoints/` directory.

---

## Overview

VALLR is a two-stage visual speech recognition system:  

```
Stage 1: Video → Phonemes (VALLR Model)
Stage 2: Phonemes → Text (LLaMA 3.2-3B with LoRA)
```

### Key Differences from Original VALLR

| Aspect | Original VALLR | This Implementation |
|--------|---------------|---------------------|
| **Dataset** | Single words (5-15 phonemes) | Continuous speech (50-89 phonemes) |
| **Data source** | Custom word dataset | Auto_AVSR LRS3 |
| **Downsampling** | Aggressive (1568→8 tokens) | Gentle (1568→392 tokens) |
| **Sequence length** | Short | Long |
| **Training complexity** | Lower | Higher |

---

## Architecture Modifications

### Problem:  Sequence Length Mismatch

The original VALLR was designed for isolated words. Auto_AVSR contains full sentences, causing CTC loss failures:  

```
Original: 1568 tokens → (stride 2,2,2,3 + AvgPool stride 8) → 8 tokens ❌
         8 tokens < 50-89 phonemes → BATCH SKIPPED

Modified: 1568 tokens → (stride 2,2 only) → 392 tokens ✅
         392 tokens > 50-89 phonemes → TRAINING WORKS
```

### Modified Downsampling (`Models/VALLR. py`)

```python
# Reduced downsampling for long sequences
self.downsampling = nn.Sequential(
    nn.Conv1d(in_channels=videomae_feature_size, out_channels=adapter_dim, 
              kernel_size=5, stride=2, padding=2),
    nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
    nn.ReLU(),

    nn.Conv1d(in_channels=adapter_dim, out_channels=adapter_dim, 
              kernel_size=3, stride=2, padding=1),
    nn.BatchNorm1d(adapter_dim, eps=1e-5, momentum=0.1, affine=True),
    nn.ReLU(),
    # Removed:  stride=2, stride=3, and AvgPool stride=8 layers
)
```

**Result:** 1568 → 784 → 392 tokens (sufficient for CTC alignment)

---

## Dataset Preparation

### Prerequisites

1. **Download LRS3 Dataset**
   ```bash
   # Contact https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html
   # Extract to:  /path/to/lrs3_raw/
   ```

2. **Process with Auto_AVSR**
   ```bash
   git clone https://github.com/mpc001/auto_avsr
   cd auto_avsr
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run preprocessing
   python preparation/lrs3.py \
       --data-dir /path/to/lrs3_raw \
       --detector retinaface \
       --dataset lrs3 \
       --seg-duration 16
   ```

3. **Expected Output Structure**
   ```
   lrs3/
   └── lrs3_video_seg16s/
       ├── trainval/
       │   ├── video_id1/
       │   │   ├── 00001.mp4
       │   │   ├── 00001.txt
       │   │   └── ...
       │   └── video_id2/
       │       └── ...
       └── test/
           └── ...  
   ```

4. **Verify Data**
   ```bash
   ls /path/to/lrs3/lrs3_video_seg16s/trainval/ | head -5
   # Should show video ID folders
   
   cat /path/to/lrs3/lrs3_video_seg16s/trainval/[video_id]/00001.txt
   # Text:  THE QUICK BROWN FOX JUMPS
   ```

---

## Training Strategy

### Quick Testing Before Full Training

Always test with a small subset first!  

#### Part 1 Test Run
```bash
python main.py \
    --mode train \
    --version V1 \
    --videos_root /path/to/lrs3/lrs3_video_seg16s \
    --save_model_path ./checkpoints/test_model.pth \
    --batch_size 4 \
    --num_workers 2 \
    --epochs 3 \
    --sample_size 0. 01  # Only 1% of data
```

**Expected output:**
```
Epoch [3/3], Training Loss: 35.99, Training Accuracy: 3.90%
Video features shape: torch.Size([4, 1568, 768])
Downsampled features shape: torch.Size([4, 256, 392])  ✅ Should be 392! 
```

**⚠️ If you see "Skipping batch" warnings:**
- Downsampling is still too aggressive
- Check `Models/VALLR.py` modifications applied correctly

#### Part 2 Test Run

```bash
# Quick test (5-10 minutes)
pip install datasets peft
python Models/Llama.py --test

# Custom subset
python Models/Llama.py --train-samples 500 --epochs 5

# Full training 
python Models/Llama.py --epochs 10
```


### Two Stage Training

**Step 1: Train Part 1 (Video → Phonemes)**
```bash
python main.py \
    --mode train \
    --version V1 \
    --videos_root /path/to/lrs3/lrs3_video_seg16s \
    --save_model_path ./checkpoints/vallr_part1.pth \
    --batch_size 8 \
    --num_workers 4 \
    --epochs 50 \
    --sample_size 1. 0
```


**Monitor Training:**
- Check `./wandb/` logs


**Step 2: Train Part 2 (Phonemes → Text)**
```bash
python Models/Llama.py
```

**Output:**
- `./llama_phonemes_to_text_lora/` - LoRA adapters
- `./llama_phonemes_to_text_lora_merged/` - Full merged model

---

## Inference

### Prerequisites

After training, you should have:
- `./checkpoints/vallr_part1.pth` (Part 1 model)
- `./llama_phonemes_to_text_lora/` (Part 2 model)

Install inference dependencies:
```bash
pip install g2p-en jiwer
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

---

### Single Video Inference

#### Video → Phonemes Only

```bash
python inference.py \
    --video /path/to/video.mp4 \
    --part1-model ./checkpoints/vallr_part1.pth \
    --part2-model ./llama_phonemes_to_text_lora \
    --part1-only
```

#### Video → Text (Full Pipeline)

```bash
python inference.py \
    --video /path/to/video.mp4 \
    --part1-model ./checkpoints/vallr_part1.pth \
    --part2-model ./llama_phonemes_to_text_lora
```


---

### Test Part 2 Model Only

Test Part 2 model independently with text input (useful for debugging):

```bash
python inference.py \
    --test-part2 "hello world" \
    --part2-model ./llama_phonemes_to_text_lora
```

**Output:**
```
============================================================
Part 2 Standalone Test (Text → Phonemes → Text)
============================================================

Input text: hello world
Phonemes:  HH AH L OW W ER L D

Generated text: hello world
Original text: hello world
============================================================
```

**How it works:**
1. Converts text to phonemes using g2p (grapheme-to-phoneme)
2. Passes phonemes through Part 2 model
3. Compares generated text with original

**Use cases:**
- Test Part 2 model before Part 1 training completes
- Debug Part 2 model performance
- Verify model loads correctly

---

### Test Set Evaluation

Process entire test folder and calculate WER:

```bash
python inference.py \
    --folder /path/to/lrs3/lrs3_video_seg16s/test \
    --part1-model ./checkpoints/vallr_part1.pth \
    --part2-model ./llama_phonemes_to_text_lora \
    --output test_results.json
```

**Output:**
```
Found 150 videos
Processing videos:   100%|████████████| 150/150 [02:30<00:00]

============================================================
INFERENCE SUMMARY
============================================================
Total videos: 150
Successfully processed: 148
Failed: 2

Word Error Rate (WER):
  Average WER: 23.45%
  Samples with ground truth: 148

Results saved to test_results.json
============================================================
```

**Results file (`test_results.json`):**
```json
{
  "summary": {
    "total_videos": 150,
    "successful": 148,
    "failed": 2,
    "average_wer": 23.45,
    "wer_samples": 148
  },
  "results": [
    {
      "video":  "/path/video1.mp4",
      "phonemes": "DH AH K AE T",
      "ground_truth": "THE CAT",
      "predicted_text": "THE CAT",
      "wer": 0.0
    },
    {
      "video": "/path/video2.mp4",
      "phonemes": "HH EH L OW W ER L D",
      "ground_truth": "HELLO WORLD",
      "predicted_text":  "HELLO WORD",
      "wer": 50.0
    }
  ]
}
```

---

### Inference Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video` | No* | - | Path to single video file |
| `--folder` | No* | - | Path to folder containing videos |
| `--test-part2` | No* | - | Test Part 2 with text input (e.g., "hello world") |
| `--part1-model` | Conditional** | - | Path to Part 1 checkpoint (. pth) |
| `--part2-model` | Conditional*** | - | Path to Part 2 LoRA model directory |
| `--output` | No | `results.json` | Output JSON file for folder processing |
| `--device` | No | `cuda` | Device to use (`cuda` or `cpu`) |
| `--part1-only` | No | `False` | Run Part 1 only (skip text generation) |
| `--max-tokens` | No | `100` | Maximum tokens to generate in Part 2 |

*One of `--video`, `--folder`, or `--test-part2` must be provided  
**Required for `--video` or `--folder` modes  
***Required unless `--part1-only` is specified

---

### Inference Examples

#### Example 1: Quick Part 1 Test
```bash
python inference.py \
    --video test.mp4 \
    --part1-model ./checkpoints/test_model.pth \
    --part2-model dummy \
    --part1-only
```

#### Example 2: Quick Part 2 Test
```bash
python inference.py \
    --test-part2 "the quick brown fox" \
    --part2-model ./llama_phonemes_to_text_lora_test
```

#### Example 3: Full Pipeline on Single Video
```bash
python inference.py \
    --video /path/to/lrs3/test/video_id/00001.mp4 \
    --part1-model ./checkpoints/vallr_part1.pth \
    --part2-model ./llama_phonemes_to_text_lora
```

#### Example 4: Batch Evaluation
```bash
python inference.py \
    --folder /path/to/lrs3/lrs3_video_seg16s/test \
    --part1-model ./checkpoints/vallr_part1.pth \
    --part2-model ./llama_phonemes_to_text_lora \
    --output final_results.json
```

#### Example 5: CPU Inference
```bash
python inference.py \
    --video test.mp4 \
    --part1-model ./checkpoints/vallr_part1.pth \
    --part2-model ./llama_phonemes_to_](#)


---

## Citation

We used the following work to built this repository: 

```bibtex
@article{thomas2025vallr,
  title={VALLR: Visual ASR Language Model for Lip Reading},
  author={Thomas, Marshall and Fish, Edward and Bowden, Richard},
  journal={arXiv preprint arXiv:2503.21408},
  year={2025}
}

@inproceedings{ma2023auto,
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={ICASSP},
  year={2023}
}
```

---

## Githubs: 

- **VALLR architecture:** https://github.com/MarshallT-99/VALLR/
- **Auto_AVSR preprocessing:** https://github.com/mpc001/auto_avsr/

---

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

**Happy Training!  🚀**
