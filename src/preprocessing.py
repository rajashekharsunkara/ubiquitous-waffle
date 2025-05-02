import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import librosa
import torch
import torchaudio
from scipy.signal import butter, filtfilt
from python_speech_features import mfcc
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, data_dir, img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        transformed = self.augmentation(image=img)
        return transformed['image']

    def prepare_dataset(self, labels_path=None, split_ratio=0.2):
        """Prepare dataset with labels if provided"""
        images = []
        labels = []
        image_paths = []

        # If labels file exists, read it
        if labels_path and os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    try:
                        processed_img = self.load_and_preprocess_image(img_path)
                        images.append(processed_img)
                        image_paths.append(img_path)
                        
                        # If labels exist, add them
                        if labels_path and os.path.exists(labels_path):
                            label = labels_df[labels_df['image_id'] == file]['label'].values[0]
                            labels.append(label)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

        # Convert to numpy arrays
        X = np.array(images)
        if labels:
            y = np.array(labels)
            # Split dataset
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=split_ratio, random_state=42, stratify=y
            )
            return X_train, X_val, y_train, y_val
        
        return X, image_paths

    def get_class_weights(self, y):
        """Calculate class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        return compute_class_weight('balanced', classes=np.unique(y), y=y)

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, frame_length=2.5, frame_overlap=0.5,
                 n_mels=128, n_mfcc=40, n_chroma=12):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_overlap = frame_overlap
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        
        # Frame parameters in samples
        self.frame_samples = int(frame_length * sample_rate)
        self.hop_samples = int(frame_length * (1 - frame_overlap) * sample_rate)
        
    def high_pass_filter(self, audio, cutoff=100.0, order=4):
        """Apply high-pass filter to remove background noise"""
        nyquist = self.sample_rate * 0.5
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, audio)
    
    def normalize_audio(self, audio):
        """Normalize audio using z-score normalization"""
        return (audio - np.mean(audio)) / np.std(audio)
    
    def extract_features(self, audio):
        """Extract all required audio features"""
        # Ensure audio is normalized
        audio = self.normalize_audio(audio)
        
        # Calculate STFT
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Calculate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Calculate MFCCs
        mfccs = mfcc(audio, self.sample_rate, numcep=self.n_mfcc)
        
        # Calculate Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # Calculate Chroma Features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, n_chroma=self.n_chroma
        )
        
        # Stack all features
        features = np.vstack([
            mel_spec_db,
            mfccs.T,
            zcr,
            chroma
        ])
        
        return features
    
    def segment_audio(self, audio):
        """Segment audio into frames with overlap"""
        frames = []
        for i in range(0, len(audio) - self.frame_samples, self.hop_samples):
            frame = audio[i:i + self.frame_samples]
            if len(frame) == self.frame_samples:  # Only keep complete frames
                frames.append(frame)
        return np.array(frames)
    
    def process_audio_file(self, file_path):
        """Process a single audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Apply high-pass filter
            audio = self.high_pass_filter(audio)
            
            # Segment audio
            frames = self.segment_audio(audio)
            
            # Extract features for each frame
            frame_features = []
            for frame in frames:
                features = self.extract_features(frame)
                frame_features.append(features)
            
            return np.array(frame_features)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def prepare_dataset(self, data_dir, labels_path=None, split_ratio=0.2):
        """Prepare dataset from audio files"""
        features = []
        labels = []
        
        # If labels file exists, read it
        if labels_path and os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            
        # Process audio files
        print("Processing audio files...")
        for root, _, files in os.walk(data_dir):
            for file in tqdm(files):
                if file.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(root, file)
                    file_features = self.process_audio_file(file_path)
                    
                    if file_features is not None:
                        features.append(file_features)
                        
                        # Get label if available
                        if labels_path and os.path.exists(labels_path):
                            label = labels_df[labels_df['file_name'] == file]['label'].values[0]
                            labels.extend([label] * len(file_features))
        
        # Convert to numpy arrays
        X = np.array(features)
        if labels:
            y = np.array(labels)
            # Split dataset
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=split_ratio, random_state=42, stratify=y
            )
            return X_train, X_val, y_train, y_val
        
        return X
    
    def get_class_weights(self, y):
        """Calculate class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        return compute_class_weight('balanced', classes=np.unique(y), y=y)