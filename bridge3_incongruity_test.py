#!/usr/bin/env python3
"""
Bridge 3 Test: Audio-Text Incongruity from Audio
==================================================
Extracts setup vs punchline audio embeddings and computes incongruity distance.

Key insight from GCACU (ChuckleNet): 
- Setup embedding vs punchline embedding → incongruity distance
- High incongruity → more unexpectedness → higher humor score

This version uses AUDIO-ONLY (no text model) to test if 
audio-based incongruity alone correlates with humor scores.

The full Bridge 3 would use: text (RoBERTa) + audio (WavLM) incongruity.
This test version: audio (WavLM) setup vs punchline incongruity.
"""
import os
import json
import numpy as np
import torch
import librosa
from pathlib import Path
from transformers import WavLMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

AUDIO_DIR = Path("/tmp/standup4ai_audio/")
PSEUDO_LABELS = "/tmp/bridge1_pseudolabels_v2.json"
SETUP_RATIO = 0.62  # From ToM's SetupPunchlineSegmenter (punchline starts at 62%)


def extract_wavlm_embedding(waveform: np.ndarray, sr: int, wavlm_model) -> np.ndarray:
    """Extract 768-dim WavLM last_hidden_state mean embedding."""
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    with torch.no_grad():
        out = wavlm_model(input_values=torch.tensor(waveform).float().unsqueeze(0).to(DEVICE))
        # last_hidden_state: (1, time, 768) → mean → (768,)
        emb = out.last_hidden_state.mean(1).squeeze().cpu().numpy()
    return emb  # 768-dim


def compute_audio_incongruity(wavlm_model, audio_path: str, n_segments: int = 20) -> dict:
    """
    Compute per-segment audio incongruity:
    - Split each segment at 62% (setup 62%, punchline 38%)
    - Extract WavLM embedding for each half
    - Incongruity = |punchline_emb - setup_emb|_1 / 768
    
    Returns dict with per-segment incongruity scores.
    """
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(waveform) / sr
    seg_duration = duration / n_segments
    
    results = []
    
    for i in range(n_segments):
        start = int(i * seg_duration * 16000)
        end = int((i + 1) * seg_duration * 16000)
        seg_wav = waveform[start:end]
        
        if len(seg_wav) < 3200:  # <200ms — too short
            results.append({"segment_idx": i, "incongruity": 0.0, "setup_emb": None, "punchline_emb": None})
            continue
        
        # Split at 62%
        split_point = int(len(seg_wav) * SETUP_RATIO)
        setup_wav = seg_wav[:split_point]
        punchline_wav = seg_wav[split_point:]
        
        # Extract embeddings
        setup_emb = extract_wavlm_embedding(setup_wav, 16000, wavlm_model)  # 768-dim
        punchline_emb = extract_wavlm_embedding(punchline_wav, 16000, wavlm_model)  # 768-dim
        
        # L1 distance normalized by embedding dimension
        incongruity = np.abs(punchline_emb - setup_emb).mean()  # scalar
        
        # Cosine similarity (alternative metric)
        cos_sim = np.dot(setup_emb, punchline_emb) / (np.linalg.norm(setup_emb) * np.linalg.norm(punchline_emb) + 1e-8)
        
        results.append({
            "segment_idx": i,
            "incongruity": float(incongruity),
            "cos_sim": float(cos_sim),
            "setup_emb_norm": float(np.linalg.norm(setup_emb)),
            "punchline_emb_norm": float(np.linalg.norm(punchline_emb)),
        })
    
    return results


def main():
    print("=" * 60)
    print("BRIDGE 3: Audio Incongruity Test")
    print("=" * 60)
    
    # Load WavLM
    print("\n[1/3] Loading WavLM...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
    wavlm.eval()
    print(f"  ✓ WavLM-base-plus loaded")
    
    # Load pseudo-labels (for comparison)
    print("\n[2/3] Loading Bridge 1 pseudo-labels...")
    with open(PSEUDO_LABELS) as f:
        pseudo = json.load(f)
    print(f"  ✓ {len(pseudo)} files loaded")
    
    # Process files
    audio_files = sorted(AUDIO_DIR.glob("*.m4a"))
    print(f"\n[3/3] Processing {len(audio_files)} files...")
    
    all_results = {}
    all_incongruities = []
    all_laughter_probs = []
    
    for af in audio_files:
        fname = af.name
        print(f"\n  {fname}")
        
        try:
            # Get pseudo-label for this file
            pseudo_data = pseudo.get(fname, {})
            laughter_probs = pseudo_data.get("laughter_probs", [])
            humor_scores = pseudo_data.get("humor_scores", [])
            
            # Compute incongruity
            incongruity_data = compute_audio_incongruity(wavlm, str(af), n_segments=20)
            
            incongruities = [d["incongruity"] for d in incongruity_data]
            cos_sims = [d["cos_sim"] for d in incongruity_data]
            
            # Use first N segments matching pseudo-label count
            n_match = min(len(incongruities), len(laughter_probs))
            matched_incongruities = incongruities[:n_match]
            matched_laughter = laughter_probs[:n_match]
            
            # Correlation
            if n_match >= 3:
                from scipy.stats import spearmanr, pearsonr
                rho, p_rho = spearmanr(matched_incongruities, matched_laughter)
                r, p_r = pearsonr(matched_incongruities, matched_laughter)
                print(f"    Segments: {n_match}")
                print(f"    Incongruity: mean={np.mean(matched_incongruities):.4f} ± {np.std(matched_incongruities):.4f}")
                print(f"    Spearman ρ={rho:.3f} (p={p_rho:.3f}), Pearson r={r:.3f} (p={p_r:.3f})")
            else:
                rho = r = 0.0
                print(f"    Too few segments ({n_match})")
            
            all_results[fname] = {
                "laughter_probs": matched_laughter,
                "humor_scores": humor_scores[:n_match],
                "incongruities": matched_incongruities,
                "cos_sims": cos_sims[:n_match],
                "spearman_rho": float(rho) if n_match >= 3 else None,
                "pearson_r": float(r) if n_match >= 3 else None,
            }
            
            all_incongruities.extend(matched_incongruities)
            all_laughter_probs.extend(matched_laughter)
            
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
    
    # Overall correlation
    print("\n" + "=" * 60)
    print("OVERALL CORRELATION: Audio Incongruity vs Laughter Probability")
    print("=" * 60)
    
    if len(all_incongruities) >= 3:
        from scipy.stats import spearmanr, pearsonr
        rho, p_rho = spearmanr(all_incongruities, all_laughter_probs)
        r, p_r = pearsonr(all_incongruities, all_laughter_probs)
        print(f"  N = {len(all_incongruities)} segments")
        print(f"  Spearman ρ = {rho:.4f} (p = {p_rho:.4f})")
        print(f"  Pearson r  = {r:.4f} (p = {p_r:.4f})")
        
        # Interpretation
        if abs(rho) < 0.1:
            interp = "Negligible — audio incongruity does NOT predict laughter"
        elif 0.1 <= abs(rho) < 0.3:
            interp = f"{'Positive' if rho > 0 else 'Negative'} weak — marginal signal"
        elif 0.3 <= abs(rho) < 0.5:
            interp = f"{'Positive' if rho > 0 else 'Negative'} moderate — useful signal"
        else:
            interp = f"{'Positive' if rho > 0 else 'Negative'} strong — powerful predictor"
        print(f"  Interpretation: {interp}")
        
        # This is based on audio-only incongruity
        # For full Bridge 3, we'd use text+audio incongruity (GCACU-style)
        print(f"\n  NOTE: This uses AUDIO-ONLY incongruity (no text).")
        print(f"  Full Bridge 3 uses RoBERTa text encoder + WavLM audio incongruity.")
        print(f"  Expected: text+audio incongruity > audio-only incongruity.")
        
        # Save
        out = {
            "overall_spearman_rho": float(rho),
            "overall_pearson_r": float(r),
            "n_segments": len(all_incongruities),
            "files": all_results,
        }
        with open("/tmp/bridge3_incongruity_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  ✓ Saved to /tmp/bridge3_incongruity_results.json")
    
    return all_results


if __name__ == "__main__":
    main()
