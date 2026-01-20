# Data/auto_avsr_dataset.py

import os
import torch
import numpy as np
from torch.utils. data import Dataset
from decord import VideoReader, cpu
from pathlib import Path
import pronouncing
import re

class AutoAVSRDataset(Dataset):
    """
    Dataset loader for Auto_AVSR preprocessed LRS3
    Compatible with original VALLR data pipeline
    No face cropping needed - Auto_AVSR already preprocessed faces
    """
    
    def __init__(self, 
                 video_dir,  # Keep same signature as VideoDataset
                 split="train",
                 phoneme_vocab=None,
                 num_frames=16,
                 frame_size=(224, 224),
                 train_val_split=0.9,  # 90% train, 10% val
                 use_pretrain=True):  # Whether to include pretrain data in training
        """
        Args: 
            video_dir: Path to lrs3 video folder (e.g., . ../lrs_combined_video_seg16s/lrs3)
            split: "train", "val", or "test"
            phoneme_vocab:  Phoneme to index mapping
            num_frames: Number of frames to sample
            frame_size: (H, W) to resize frames
            train_val_split: Fraction of trainval to use for training (rest is validation)
            use_pretrain: If True, adds pretrain data to train split (153K+ videos)
        """
        # Map split names to Auto_AVSR structure
        # Both train and val load from "trainval", then we split them
        split_map = {
            "train": "trainval",
            "val": "trainval",  
            "test": "test"
        }
        avsr_split = split_map. get(split, split)
        
        self.video_root = Path(video_dir)
        self.video_dir = self. video_root / avsr_split
        self.train_val_split = train_val_split
        self.use_pretrain = use_pretrain and (split == "train")  # Only add pretrain to train
        
        # Automatically derive text directory by replacing "video" with "text"
        self.text_dir = Path(str(self.video_root).replace("video", "text")) / avsr_split
        
        self.split = split
        self. phoneme_vocab = phoneme_vocab
        self.num_frames = num_frames
        self. frame_size = frame_size
        
        # Verify directories exist
        if not self.video_dir.exists():
            raise ValueError(f"Video directory not found:  {self.video_dir}")
        if not self.text_dir.exists():
            raise ValueError(f"Text directory not found: {self.text_dir}")
        
        # Collect all video-text pairs
        self.video_paths = []
        self.labels = []
        self._build_dataset()
        
        # Add pretrain data if requested (only for train split)
        if self.use_pretrain:
            print(f"Adding pretrain data to training set...")
            pretrain_video_dir = self.video_root / "pretrain"
            pretrain_text_dir = Path(str(self.video_root).replace("video", "text")) / "pretrain"
            
            if pretrain_video_dir.exists() and pretrain_text_dir.exists():
                pretrain_count_before = len(self.video_paths)
                self._build_dataset_from_dir(pretrain_video_dir, pretrain_text_dir)
                pretrain_added = len(self.video_paths) - pretrain_count_before
                print(f"✅ Added {pretrain_added} pretrain videos")
        
        # Apply train/val split if loading from trainval
        if avsr_split == "trainval" and split in ["train", "val"]:
            total_samples = len(self.video_paths)
            split_idx = int(total_samples * self.train_val_split)
            
            if split == "train":
                # Take first 90%
                self.video_paths = self.video_paths[:split_idx]
                self.labels = self.labels[:split_idx]
            else:  # val
                # Take last 10%
                self.video_paths = self.video_paths[split_idx:]
                self.labels = self.labels[split_idx:]
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No samples found in {split} split!")
        
        print(f"Loaded {len(self.video_paths)} videos from {split} split in {video_dir}.")
    
    def _build_dataset(self):
        """Build list of video paths and labels from current video_dir and text_dir"""
        self._build_dataset_from_dir(self.video_dir, self.text_dir)
    
    def _build_dataset_from_dir(self, video_dir, text_dir):
        """Build dataset from specified directories"""
        # Get all speaker ID folders
        speaker_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
        
        for speaker_dir in speaker_dirs: 
            speaker_id = speaker_dir.name
            text_speaker_dir = text_dir / speaker_id
            
            if not text_speaker_dir.exists():
                continue
            
            # Get all . mp4 files
            video_files = sorted(speaker_dir. glob("*.mp4"))
            
            for video_file in video_files:
                text_file = text_speaker_dir / f"{video_file.stem}.txt"
                
                if text_file.exists() and text_file.stat().st_size > 0:
                    self.video_paths.append(str(video_file))
                    # Store text file path as label (will convert to phonemes in __getitem__)
                    self.labels.append(str(text_file))
    
    def __len__(self):
        return len(self. video_paths)
    
    def __getitem__(self, idx):
        """Load video and convert text to phonemes"""
        
        # Retry logic if sample is invalid
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            video_path = self.video_paths[idx]
            text_path = self.labels[idx]
            
            # Load video
            video = self._load_video(video_path)
            if video is None:
                retries += 1
                idx = (idx + 1) % len(self.video_paths)
                continue
            
            # Load text and convert to phonemes
            text = self._load_text(text_path)
            if not text: 
                retries += 1
                idx = (idx + 1) % len(self.video_paths)
                continue
            
            phoneme_indices = self._text_to_phonemes(text)
            if len(phoneme_indices) == 0:
                retries += 1
                idx = (idx + 1) % len(self.video_paths)
                continue
            
            # Convert to tensors (same format as original VideoDataset)
            video_tensor = torch. tensor(video).permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            phoneme_tensor = torch. tensor(phoneme_indices, dtype=torch.long)
            
            return video_tensor, phoneme_tensor
        
        raise ValueError(f"Exceeded max retries for video at index {idx}")
    
    def _load_video(self, video_path):
        """Load and preprocess video (no face cropping - already done by Auto_AVSR)"""
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        except Exception as e:
            print(f"Error loading video:  {video_path}. Error: {e}")
            return None
        
        frame_count = len(vr)
        if frame_count < self.num_frames:
            print(f"Warning: Not enough frames in video {video_path}. Skipping.")
            return None
        
        # Sample indices uniformly
        sample_indices = np.linspace(0, frame_count - 1, self.num_frames).astype(int)
        
        frames = []
        for idx in sample_indices:
            frame = vr[idx].asnumpy()
            # Resize if needed
            import cv2
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        
        if len(frames) < self.num_frames:
            return None
        
        return np.array(frames)  # (T, H, W, C)
    
    def _load_text(self, text_path):
        """Load text from file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f. read().strip()
            
            if not text:
                return None
            
            # Remove "Text:" prefix if exists
            if text.startswith("Text:"):
                text = text. replace("Text:", "").strip()
            
            return text. upper().strip() if text else None
        except:
            return None
    
    def _text_to_phonemes(self, text):
        """Convert text to phoneme indices"""
        words = text.split()
        phoneme_indices = []
        
        for word in words:
            phonemes = self._get_phonemes(word)
            for phoneme in phonemes:
                if phoneme in self.phoneme_vocab:
                    phoneme_indices. append(self.phoneme_vocab[phoneme])
        
        return phoneme_indices
    
    def _get_phonemes(self, word):
        """Get phonemes for a word using pronouncing library"""
        # Remove punctuation
        word = re.sub(r'[^\w\s]', '', word).lower()
        
        if not word:  
            return []
        
        phones = pronouncing.phones_for_word(word)
        
        if phones:
            # Take first pronunciation, remove stress markers
            phoneme_list = [re.sub(r'\d+', '', p) for p in phones[0].split()]
            return phoneme_list
        else: 
            return []


def load_and_preprocess_video(video_path, num_frames):
    """
    Compatibility function to match original dataset. py signature
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    except Exception as e:  
        print(f"Error loading video: {video_path}. Error: {e}")
        return None

    frame_count = len(vr)
    if frame_count < num_frames: 
        return None

    sample_indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in sample_indices: 
        frame = vr[idx].asnumpy()
        frames.append(frame)

    if len(frames) < num_frames:
        return None

    return np. array(frames)


def get_phonemes(sentence):
    """
    Compatibility function to match original dataset.py signature
    """
    words = sentence.split()
    phoneme_list = []
    
    for word in words: 
        word = re.sub(r'[^\w\s]', '', word).lower()
        if not word:
            continue
        
        phones = pronouncing. phones_for_word(word)
        if phones:
            phoneme_list.extend([re. sub(r'\d+', '', p) for p in phones[0].split()])
    
    return phoneme_list