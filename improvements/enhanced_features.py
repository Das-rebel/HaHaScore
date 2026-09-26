#!/usr/bin/env python3
"""
Enhanced Feature Extraction for Cascade Gate v7
================================================
Adds advanced prosody, text, and cross-modal features.
"""
import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy import signal
from scipy.stats import skew, kurtosis


def extract_enhanced_prosody(audio_samples, sr=16000, n_segments=20):
    """
    Extract enhanced prosody features per segment.
    
    Args:
        audio_samples: np.ndarray, shape (n_samples,)
        sr: int, sample rate
        n_segments: int, number of segments
        
    Returns:
        np.ndarray, shape (n_segments, n_features)
    """
    segment_len = len(audio_samples) // n_segments
    features = []
    
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len if i < n_segments - 1 else len(audio_samples)
        seg = audio_samples[start:end]
        
        if len(seg) == 0:
            # Empty segment - pad with zeros
            features.append(np.zeros(30, dtype=np.float32))
            continue
            
        # F0 contour analysis
        try:
            f0 = librosa.yin(seg, fmin=50, fmax=500, sr=sr)
            f0 = np.nan_to_num(f0)
        except:
            f0 = np.zeros(len(seg))
        
        # Amplitude envelope
        try:
            rms = librosa.feature.rms(y=seg)
            rms_smooth = np.convolve(rms[0], np.ones(5)/5, mode='same')
        except:
            rms_smooth = np.zeros(len(seg) // 512 + 1)
        
        # Zero crossing rate
        try:
            zcr = librosa.feature.zero_crossing_rate(seg)
        except:
            zcr = np.zeros(len(seg) // 512 + 1)
        
        # MFCCs
        try:
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        except:
            mfcc = mfcc_delta = mfcc_delta2 = np.zeros((13, len(seg) // 512 + 1))
        
        # Spectral features
        try:
            centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr)
        except:
            centroid = bandwidth = rolloff = np.zeros((1, len(seg) // 512 + 1))
        
        # Formant-like features (simplified)
        try:
            stft = np.abs(librosa.stft(seg))
            power_spectrum = np.mean(stft, axis=1)
            if len(power_spectrum) > 10:
                formant_peaks = signal.find_peaks(power_spectrum, distance=5)[0]
                formant_features = formant_peaks[:3] if len(formant_peaks) >= 3 else np.pad(formant_peaks, (0, max(0, 3-len(formant_peaks))))
            else:
                formant_features = np.zeros(3)
        except:
            formant_features = np.zeros(3)
        
        # Jitter and shimmer approximations
        f0_diff = np.diff(f0) if len(f0) > 1 else np.array([0])
        jitter = np.mean(np.abs(f0_diff))
        amp_diff = np.diff(rms_smooth) if len(rms_smooth) > 1 else np.array([0])
        shimmer = np.mean(np.abs(amp_diff))
        
        # Pause patterns
        pause_threshold = np.percentile(rms_smooth, 20) if len(rms_smooth) > 0 else 0
        pauses = (rms_smooth < pause_threshold).mean() if len(rms_smooth) > 0 else 0
        
        # Speech rate approximation
        speech_rate = np.sum(rms_smooth > pause_threshold) / max(len(rms_smooth), 1) if len(rms_smooth) > 0 else 0
        
        # Skewness and kurtosis
        f0_skew = skew(f0) if len(f0) > 1 else 0
        f0_kurt = kurtosis(f0) if len(f0) > 3 else 0
        
        # Aggregate features (30 features)
        feat = [
            np.mean(f0), np.std(f0), np.median(f0), np.percentile(f0, 90) - np.percentile(f0, 10),
            np.mean(rms_smooth), np.std(rms_smooth), np.max(rms_smooth),
            np.mean(zcr), np.std(zcr),
            np.mean(mfcc), np.std(mfcc), np.mean(mfcc_delta), np.std(mfcc_delta),
            np.mean(mfcc_delta2), np.std(mfcc_delta2),
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            formant_features[0] if len(formant_features) > 0 else 0,
            formant_features[1] if len(formant_features) > 1 else 0,
            formant_features[2] if len(formant_features) > 2 else 0,
            jitter, shimmer, pauses, speech_rate,
            f0_skew, f0_kurt,
            np.mean(f0), np.std(f0),  # extra f0 stats
        ]
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def extract_text_context_features(text_features, window_size=3):
    """
    Extract contextual text features from text embeddings.
    Returns 12 statistical features per segment based on L2 norm of text vectors in window.
    
    Args:
        text_features: np.ndarray, shape (n_segments, 768)
        window_size: int, context window size
        
    Returns:
        np.ndarray, shape (n_segments, 12)
    """
    n_segments = len(text_features)
    # Compute L2 norm per segment
    norms = np.linalg.norm(text_features, axis=1)  # (n_segments,)
    context_features = []
    
    for i in range(n_segments):
        start = max(0, i - window_size)
        end = min(n_segments, i + window_size + 1)
        window_norms = norms[start:end]
        
        if len(window_norms) == 0:
            stats = [0.0] * 12
        else:
            stats = [
                np.mean(window_norms),
                np.std(window_norms),
                np.min(window_norms),
                np.max(window_norms),
                np.median(window_norms),
                np.ptp(window_norms),  # peak-to-peak
                skew(window_norms) if len(window_norms) > 2 else 0.0,
                kurtosis(window_norms) if len(window_norms) > 3 else 0.0,
                np.sqrt(np.mean(window_norms**2)),  # RMS
                np.sum(window_norms),  # energy
                np.sum(np.diff(window_norms) != 0) if len(window_norms) > 1 else 0,  # zero crossings approx
                np.percentile(window_norms, 75) - np.percentile(window_norms, 25),  # IQR
            ]
        context_features.append(stats)
    
    return np.array(context_features, dtype=np.float32)


def extract_cross_modal_features(text_features, audio_features):
    """
    Extract cross-modal alignment features.
    
    Args:
        text_features: np.ndarray, shape (n_segments, 768)
        audio_features: np.ndarray, shape (n_segments, 791)
        
    Returns:
        np.ndarray, shape (n_segments, 16)
    """
    n_segments = len(text_features)
    cross_features = []
    
    # Normalize features separately
    text_norm = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
    audio_norm = audio_features / (np.linalg.norm(audio_features, axis=1, keepdims=True) + 1e-8)
    
    for i in range(n_segments):
        # Cosine similarity between text and audio (use min dim)
        min_dim = min(len(text_norm[i]), len(audio_norm[i]))
        similarity = np.dot(text_norm[i][:min_dim], audio_norm[i][:min_dim])
        
        # Local alignment
        text_local = text_norm[max(0, i-1):min(n_segments, i+2)]
        audio_local = audio_norm[max(0, i-1):min(n_segments, i+2)]
        local_sims = []
        for t, a in zip(text_local, audio_local):
            md = min(len(t), len(a))
            local_sims.append(np.dot(t[:md], a[:md]))
        local_sim = np.mean(local_sims)
        
        # Temporal dynamics
        text_diff = np.mean(np.diff(text_norm, axis=0), axis=0) if n_segments > 1 else np.zeros_like(text_norm[0])
        audio_diff = np.mean(np.diff(audio_norm, axis=0), axis=0) if n_segments > 1 else np.zeros_like(audio_norm[0])
        md = min(len(text_diff), len(audio_diff))
        diff_sim = np.dot(text_diff[:md], audio_diff[:md])
        
        # Energy and context
        text_energy = np.mean(np.linalg.norm(text_features[i]))
        audio_energy = np.mean(np.linalg.norm(audio_features[i]))
        
        # Alignment features
        cross_features.append([
            similarity,
            local_sim,
            diff_sim,
            text_energy,
            audio_energy,
            np.std(text_features[i]),
            np.std(audio_features[i]),
            np.mean(text_features[i]),
            np.mean(audio_features[i]),
            np.linalg.norm(text_features[i]),
            np.linalg.norm(audio_features[i]),
            np.max(text_features[i]),
            np.max(audio_features[i]),
            np.min(text_features[i]),
            np.min(audio_features[i]),
            np.abs(similarity - local_sim),
        ])
    
    return np.array(cross_features, dtype=np.float32)


def build_enhanced_features(text_features, audio_features, audio_samples=None, sr=16000):
    """
    Build enhanced feature set.
    
    Args:
        text_features: np.ndarray, shape (n_segments, 768)
        audio_features: np.ndarray, shape (n_segments, 791)
        audio_samples: np.ndarray, raw audio samples (optional)
        sr: int, sample rate
        
    Returns:
        enhanced_text (text_features + context), enhanced_audio (audio + prosody), enhanced_cross
    """
    # Text: original + context stats
    text_context = extract_text_context_features(text_features)
    enhanced_text = np.concatenate([text_features, text_context], axis=1)
    
    # Audio: original + prosody
    if audio_samples is not None:
        enhanced_prosody = extract_enhanced_prosody(audio_samples, sr=sr, n_segments=len(text_features))
        enhanced_audio = np.concatenate([audio_features, enhanced_prosody], axis=1)
    else:
        enhanced_audio = audio_features
    
    # Cross-modal
    enhanced_cross = extract_cross_modal_features(text_features, audio_features)
    
    return enhanced_text, enhanced_audio, enhanced_cross


if __name__ == "__main__":
    # Example usage
    text_features = np.random.randn(20, 768).astype(np.float32)
    audio_features = np.random.randn(20, 791).astype(np.float32)
    audio_samples = np.random.randn(16000 * 60).astype(np.float32)
    
    enhanced_text, enhanced_audio, enhanced_cross = build_enhanced_features(
        text_features, audio_features, audio_samples
    )
    
    print(f"Text: {enhanced_text.shape}")          # (20, 768+12) = (20, 780)
    print(f"Audio: {enhanced_audio.shape}")       # (20, 791+30) = (20, 821)
    print(f"Cross-modal: {enhanced_cross.shape}") # (20, 16)
