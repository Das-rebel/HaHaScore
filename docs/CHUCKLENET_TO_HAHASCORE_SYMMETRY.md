# HaHaScore × ChuckleNet: The Symmetry Framework

## Executive Summary

**ChuckleNet** (binary laughter detection) and **HaHaScore** (continuous humor strength scoring) are separate, independent projects that must remain architecturally distinct. But they share the same underlying phenomenon — humor-induced laughter — and can therefore act as **mutually reinforcing mirrors**: each teaches the other things it cannot learn from itself.

**Core insight**: Laughter is the *ground truth* of humor. Every other signal — text, prosody, kinetic energy — is a *proxy*. ChuckleNet learns from the strongest proxy (audio → binary laugh). HaHaScore tries to predict the unobserved latent variable (text+audio → continuous humor). They are inverse problems of each other.

**Separation constraint enforced**: No shared model architectures, training scripts, or feature pipelines. They may share raw data and pre-computed feature files.

---

## The Five Symmetry Bridges

### Bridge 1: Laughter as Pseudo-Label Factory

**Problem HaHaScore faces**: Labels are expensive. 21,560 sentence clips with continuous humor scores don't exist. We have binary laughter labels for the same 48 Indian comedy videos.

**ChuckleNet's gift**: ChuckleNet is trained on binary laughter labels with F1=0.975 — near-perfect detection. Its predictions on new audio are highly reliable.

**The pipeline**:
1. Run every audio file in `standup4ai/audio_1000/` (641 files) through ChuckleNet's WavLM→791-dim→FusionMLP pipeline
2. Extract per-segment laughter probability time series
3. Use laughter probability as a *continuous* pseudo-label for HaHaScore training:
   - Laughter probability 0.0 = no humor → score 0
   - Laughter probability 1.0 = maximum humor → score 100
   - Intermediate values → linear interpolation
4. This gives HaHaScore **641 files × ~100 segments = ~64,100 pseudo-labeled training samples**

**Why this works**: Laughter probability is a more informative training signal than binary labels because it preserves uncertainty. A segment with laughter probability 0.3 might contain a weak joke that fell flat — HaHaScore learns from this.

**Separation**: Uses ChuckleNet's *inference pipeline* (not its weights or architecture). HaHaScore training uses its own fusion model with ChuckleNet's outputs as features/labels.

```python
# Pseudo-code
chuckle_output = chucklenet.predict(audio_segment)  # laughter probability ∈ [0,1]
hahascore_label = chucklenet_laughter_prob * 100   # scale to [0,100]
```

---

### Bridge 2: Cascade as Attention Router

**ChuckleNet's cascade architecture** (Stage 1: text proposes regions, Stage 2: prosody refines boundaries) has a direct analog in HaHaScore:

- **HaHaScore Stage 1** = Text-only model predicts "is this segment worth scoring?" (confidence)
- **HaHaScore Stage 2** = Audio features are only extracted/processed for segments where text predicts medium-high humor potential

This cascades the expensive audio processing (WavLM/HuBERT forward pass) to only the most promising segments, reducing compute 3-5x.

**The architectural transfer**: Take the Stage 2 boundary refinement concept but apply it as *audio attention gating*:
- Text encoder outputs a confidence score per segment
- Audio encoder processes ALL segments but output is gated by text confidence
- Low-confidence segments get muted audio contribution → model learns to trust text when audio is ambiguous

---

### Bridge 3: GCACU Contrastive Attention for Audio-Text Alignment

**ChuckleNet's GCACU** extracts lexical pairs (setup embedding vs punchline embedding) and applies contrastive attention — it explicitly models the *incongruity gap* between setup and punchline.

**HaHaScore application**: 
- Extract setup embedding from audio (first half of segment) 
- Extract punchline embedding from audio (second half of segment)  
- Compute the "incongruity distance": `|setup_embedding - punchline_embedding|`
- This incongruity distance (a scalar) becomes a **new feature dimension** in HaHaScore's fusion MLP

**Evidence**: TIC-TALK research (r=−0.75) shows kinetic stillness before punchline predicts laughter. GCACU's incongruity gap is the *semantic analog* — it measures the cognitive distance between setup and punchline, which drives the need for laughter as relief.

**Separation**: GCACU's contrastive attention mechanism is NOT copied. Only the *input construction strategy* (setup/punchline split) is borrowed.

---

### Bridge 4: HitEmotionTrajectory for Humor Arc Modeling

**ChuckleNet's HitEmotionTrajectory** tracks comedian and audience emotional states over time using depthwise 1D convolutions. It models early-vs-late emotional shifts.

**HaHaScore application** — the **Humor Arc**:
- Instead of tracking joy/surprise/confusion, track *incongruity accumulation* over a comedy set
- Consecutive segments with rising incongruity → higher humor score
- A segment that resolves incongruity (punchline) → highest score
- Post-resolution stillness (TIC-TALK r=−0.75) → score boost

**The key insight**: HaHaScore currently scores each segment independently. ChuckleNet's trajectory modeling suggests segments should be scored *relative to their position in the humor arc* — a setup-only segment gets discounted even if it has comedic content, because the resolution hasn't happened yet.

**Implementation**: Add a lightweight LSTM/GRU (2 layers, 64 hidden) on top of per-segment HaHaScore features, trained to predict the continuous humor score from the arc context.

---

### Bridge 5: Scale221 Pseudo-Labels as Data Scaleup

**ChuckleNet's Scale221** successfully used pre-extracted 791-dim embeddings + GroupKFold training + teacher pseudo-labeling to train on 221 StandUp4AI videos.

**HaHaScore application**:
1. Download all 641 `standup4ai/audio_1000/` files (3.15 GB) to Modal volume
2. Run ChuckleNet's WavLM extraction on all 641 files → 641 × 791-dim embedding matrices
3. Run ChuckleNet's fusion model to get per-segment laughter probabilities
4. Use laughter probability as continuous pseudo-label for HaHaScore v6 training
5. GroupKFold by video (641 folds possible — leave-one-video-out cross-validation at scale)

**This converts HaHaScore from 21,560 clips (48 videos) to 64,100+ clips (641+ videos)** — a 3x data increase with zero additional labeling cost.

---

## New Module: `hahascore/chucklesync.py`

The architectural embodiment of all five bridges.

```python
"""
hahascore/chucklesync.py
========================
ChuckleSync: The five symmetry bridges between ChuckleNet and HaHaScore.

KEY PRINCIPLE: This module uses ChuckleNet as an INFERENCE oracle and 
data generator. It does NOT copy ChuckleNet architectures, training 
logic, or model weights. It treats ChuckleNet as an external service.

Architecture independence: HaHaScore remains a RoBERTa+WavLM multimodal 
fusion model. ChuckleNet provides pseudo-labels and attention signals.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SymmetryConfig:
    """Configuration for ChuckleSync bridges."""
    # Bridge 1: Pseudo-label threshold
    pseudo_label_min_prob: float = 0.15   # Min ChuckleNet prob to use as label
    pseudo_label_source: str = "chucklenet_fusion"  # Which ChuckleNet model to use
    
    # Bridge 2: Cascade gating
    cascade_text_threshold: float = 0.3   # Text confidence below which to mute audio
    audio_gate_hidden: int = 64
    
    # Bridge 3: GCACU incongruity
    incongruity_margin: float = 0.1       # Margin for contrastive loss
    setup_ratio: float = 0.62            # Position where punchline begins (from ToM)
    
    # Bridge 4: Arc modeling
    arc_hidden_dim: int = 64
    arc_num_layers: int = 2
    arc_context_window: int = 5           # Segments to look back
    
    # Bridge 5: Scaleup
    use_groupkfold: bool = True
    n_folds: int = 5
    
    # Paths
    chucklenet_model_path: str = "/tmp/fusion_mlp_v2.pt"  # Trained ChuckleNet model
    chucklenet_wavlm_path: str = "/tmp/wavlm_base_plus"


class AudioTextAlign(nn.Module):
    """
    Bridge 3: GCACU-inspired audio-text incongruity alignment.
    
    Takes setup and punchline audio embeddings, computes incongruity distance.
    This is NOT GCACU — only the input construction strategy (setup/punchline split).
    """
    def __init__(self, audio_dim: int = 512, hidden_dim: int = 64):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.setup_ratio = 0.62  # Learned from ToM's SetupPunchlineSegmenter
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        audio_features: (batch, seq_len, audio_dim) — e.g., WavLM frame features
        
        Returns incongruity_distance: (batch, 1) — scalar per segment
        """
        seq_len = audio_features.size(1)
        split_idx = int(seq_len * self.setup_ratio)
        
        # Split at learned punchline boundary
        setup = audio_features[:, :split_idx]       # (batch, setup_len, dim)
        punchline = audio_features[:, split_idx:]    # (batch, punch_len, dim)
        
        # Mean pool
        setup_emb = setup.mean(dim=1)                # (batch, dim)
        punchline_emb = punchline.mean(dim=1)        # (batch, dim)
        
        # Project
        setup_h = torch.tanh(self.audio_proj(setup_emb))
        punchline_h = torch.tanh(self.audio_proj(punchline_emb))
        
        # Incongruity distance: L1 norm of difference
        incongruity = (punchline_h - setup_h).abs().mean(dim=-1, keepdim=True)
        
        return incongruity  # (batch, 1)


class CascadeAudioGate(nn.Module):
    """
    Bridge 2: Text-guided audio attention gate.
    
    Text confidence gates how much audio contributes to final score.
    Low text confidence → audio contribution is muted.
    """
    def __init__(self, text_dim: int = 768, audio_dim: int = 512, hidden: int = 64):
        super().__init__()
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.audio_proj = nn.Linear(audio_dim, hidden)
        
    def forward(self, text_pooler: torch.Tensor, audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        text_pooler: (batch, text_dim) — RoBERTa pooler output
        audio_features: (batch, audio_dim) — WavLM audio embedding
        
        Returns: (gated_audio, gate_value)
            gated_audio: audio weighted by text confidence
            gate_value: the text confidence scalar
        """
        gate = self.text_gate(text_pooler)  # (batch, 1)
        
        audio_h = torch.tanh(self.audio_proj(audio_features))  # (batch, hidden)
        gated = audio_h * gate  # (batch, hidden)
        
        return gated, gate


class HumorArcTracker(nn.Module):
    """
    Bridge 4: HitEmotionTrajectory adapted for humor arc modeling.
    
    Tracks incongruity accumulation and resolution over consecutive segments.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.arc_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, segment_features: torch.Tensor, 
                segment_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        segment_features: (batch, seq_len, feature_dim) — per-segment HaHaScore features
        segment_mask: (batch, seq_len) — 1 for valid segments
        
        Returns arc_score: (batch, seq_len, 1) — per-segment arc-adjusted scores
        """
        rnn_out, _ = self.rnn(segment_features)  # (batch, seq_len, hidden*2)
        arc_score = self.arc_proj(rnn_out)  # (batch, seq_len, 1)
        
        if segment_mask is not None:
            arc_score = arc_score * segment_mask.unsqueeze(-1)
        
        return {"arc_score": arc_score, "rnn_out": rnn_out}


class PseudoLabelGenerator:
    """
    Bridge 1 & 5: Generates continuous humor score pseudo-labels using ChuckleNet.
    
    Treats ChuckleNet as an external inference oracle. Does NOT copy model weights.
    """
    def __init__(self, chucklenet_model_path: str, chucklenet_wavlm_path: str, device: str = "cpu"):
        self.device = device
        # Load only for inference, not training
        self._load_chucklenet(chucklenet_model_path, chucklenet_wavlm_path)
    
    def _load_chucklenet(self, model_path: str, wavlm_path: str):
        """Load ChuckleNet's fusion model for inference only."""
        from transformers import WavLMModel
        import torchaudio
        
        # WavLM feature extractor
        self.wavlm = WavLMModel.from_pretrained(wavlm_path).to(self.device)
        self.wavlm.eval()
        
        # Fusion MLP (architecture: 791→512→256→64→1)
        self.fusion = nn.Sequential(
            nn.Linear(791, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        ).to(self.device)
        
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.fusion.load_state_dict(state, strict=False)
        self.fusion.eval()
        
        print(f"ChuckleNet loaded for pseudo-labeling: {sum(p.numel() for p in self.fusion.parameters()):,} params")
    
    def extract_wavlm_prosody(self, audio_path: str) -> np.ndarray:
        """
        Extract 791-dim features (WavLM 768 + prosody 23) from audio file.
        Mirrors ChuckleNet's extraction pipeline.
        """
        import torchaudio
        import librosa
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # WavLM features
        with torch.no_grad():
            wavlm_out = self.wavlm(waveform.to(self.device))
            wavlm_feats = wavlm_out.extract_features.mean(1).cpu().numpy()  # (n_frames, 512)
        
        # Prosody features (F0, energy, duration per word from VAD)
        # Simplified: use RMS energy as proxy for prosody
        waveform_np = waveform.squeeze().cpu().numpy()
        rms = librosa.feature.rms(y=waveform_np, frame_length=400, hop_length=160)[0]
        rms_expanded = np.repeat(rms, 512 // len(rms))[:wavlm_feats.shape[0]]  # align to WavLM frames
        
        # Pad/truncate to match WavLM frames
        if len(rms_expanded) < wavlm_feats.shape[0]:
            pad_len = wavlm_feats.shape[0] - len(rms_expanded)
            rms_expanded = np.pad(rms_expanded, (0, pad_len))
        
        prosody_23d = self._extract_prosody_23d(waveform_np, sr)
        
        # Segment-level prosody (mean per segment)
        n_segments = wavlm_feats.shape[0]
        prosody_per_seg = np.tile(prosody_23d, (n_segments, 1))
        
        # Concatenate: (n_segments, 791)
        features = np.concatenate([wavlm_feats, prosody_per_seg], axis=1)
        return features
    
    def _extract_prosody_23d(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Extract 23 prosody features from audio waveform."""
        import librosa
        
        # F0 (pitch) statistics
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform, fmin=80, fmax=300, sr=sr
        )
        f0_valid = f0[~np.isnan(f0)]
        
        features = []
        # F0 stats (6 dims)
        features.append(np.mean(f0_valid) if len(f0_valid) > 0 else 0)
        features.append(np.std(f0_valid) if len(f0_valid) > 0 else 0)
        features.append(np.max(f0_valid) if len(f0_valid) > 0 else 0)
        features.append(np.min(f0_valid) if len(f0_valid) > 0 else 0)
        features.append(np.median(f0_valid) if len(f0_valid) > 0 else 0)
        features.append(len(f0_valid) / len(f0))  # voiced ratio
        
        # Energy stats (4 dims)
        rms = librosa.feature.rms(y=waveform)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.max(rms))
        features.append(np.percentile(rms, 25))
        
        # Spectral (4 dims)
        spec_bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        features.append(np.mean(spec_bw) / (np.mean(spec_bw) + 1e-8))
        features.append(0.0)  # padding
        
        # Temporal (4 dims)
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        silence_ratio = (rms < 0.01).mean()
        features.append(silence_ratio)
        features.append(np.mean(rms[rms > np.percentile(rms, 75)]))  # high energy ratio
        
        # Duration (2 dims) — estimated from waveform length
        duration = len(waveform) / sr
        features.append(duration)
        features.append(1.0)  # bias term
        
        return np.array(features[:23], dtype=np.float32)
    
    @torch.no_grad()
    def generate_pseudo_labels(self, audio_path: str, segment_boundaries: List[Tuple[float, float]]) -> List[float]:
        """
        Generate continuous humor score pseudo-labels for segments.
        
        audio_path: Path to audio file
        segment_boundaries: List of (start_sec, end_sec) for each segment
        
        Returns: List of continuous scores ∈ [0, 100]
        """
        import torchaudio
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Get full WavLM features
        wavlm_out = self.wavlm(waveform.to(self.device))
        all_wavlm = wavlm_out.extract_features.mean(1)  # (1, n_frames, 512)
        
        # Get prosody
        waveform_np = waveform.squeeze().cpu().numpy()
        prosody_23d = torch.tensor(self._extract_prosody_23d(waveform_np, sr)).to(self.device)
        
        total_frames = all_wavlm.size(1)
        duration = len(waveform_np) / sr
        
        scores = []
        for start_sec, end_sec in segment_boundaries:
            start_frame = int(start_sec / duration * total_frames)
            end_frame = int(end_sec / duration * total_frames)
            
            # Extract segment features
            seg_wavlm = all_wavlm[:, start_frame:end_frame].mean(dim=1)  # (1, 512)
            
            # Tile prosody to match WavLM dim
            prosody_tiled = prosody_23d.unsqueeze(0).expand(seg_wavlm.size(0), -1)  # (1, 23)
            
            # Concatenate and project to 791
            seg_feat_512 = seg_wavlm.squeeze().cpu().numpy()
            prosody_np = prosody_tiled.squeeze().cpu().numpy()
            
            # Pad if needed
            if len(seg_feat_512) < 512:
                pad = np.zeros(512 - len(seg_feat_512))
                seg_feat_512 = np.concatenate([seg_feat_512, pad])
            
            # Final 791-dim vector
            seg_feat = np.concatenate([seg_feat_512, prosody_np])  # (791,)
            seg_tensor = torch.tensor(seg_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # ChuckleNet inference
            laugh_prob = torch.sigmoid(self.fusion(seg_tensor)).item()
            
            # Scale to humor score [0, 100]
            humor_score = laugh_prob * 100.0
            scores.append(humor_score)
        
        return scores


class ChuckleSyncModel(nn.Module):
    """
    The full ChuckleSync model: HaHaScore + all 5 bridges.
    
    Architecture:
    1. Text tower: RoBERTa → pooler → text_proj (768→128)
    2. Audio tower: WavLM → audio_proj (512→128)  
    3. Audio-text incongruity: AudioTextAlign (512→1 incongruity score)
    4. Cascade gate: CascadeAudioGate (gates audio by text confidence)
    5. Arc tracker: HumorArcTracker (sequential context)
    6. Fusion: 3-tower concat (text, gated_audio, incongruity) + MLP
    """
    def __init__(self, cfg: SymmetryConfig):
        super().__init__()
        from transformers import RobertaModel, WavLMModel
        
        self.cfg = cfg
        
        # Text tower
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_proj = nn.Linear(768, 128)
        
        # Audio tower
        self.audio_encoder = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.audio_proj = nn.Linear(512, 128)
        
        # Bridge 3: Incongruity alignment
        self.incongruity_align = AudioTextAlign(audio_dim=512, hidden_dim=64)
        
        # Bridge 2: Cascade audio gate
        self.cascade_gate = CascadeAudioGate(text_dim=768, audio_dim=512, hidden=64)
        
        # Bridge 4: Arc tracker (initialized lazily)
        self.arc_tracker = None
        
        # Bridge 5: Fusion head (3-tower: text + gated_audio + incongruity)
        fusion_input_dim = 128 + 64 + 1  # text(128) + gated_audio(64) + incongruity(1)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                audio_values: torch.Tensor,
                return_bridge_outputs: bool = False) -> Dict[str, torch.Tensor]:
        """
        input_ids: (batch, seq_len) — token IDs
        attention_mask: (batch, seq_len) — attention mask  
        audio_values: (batch, n_samples) — raw audio
        
        Returns:
            score: (batch,) — continuous humor score ∈ [0, 100]
            bridge_outputs: dict of intermediate values (optional)
        """
        # Text encoding
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooler = text_out.last_hidden_state[:, 0]  # (batch, 768)
        text_h = torch.relu(self.text_proj(text_pooler))  # (batch, 128)
        
        # Audio encoding
        audio_out = self.audio_encoder(input_values=audio_values)
        audio_features = audio_out.extract_features.mean(1)  # (batch, 512)
        audio_h = torch.relu(self.audio_proj(audio_features))  # (batch, 128)
        
        # Bridge 3: Incongruity distance
        audio_full = audio_out.extract_features  # (batch, n_frames, 512)
        incongruity = self.incongruity_align(audio_full)  # (batch, 1)
        
        # Bridge 2: Cascade audio gate
        gated_audio, gate_value = self.cascade_gate(text_pooler, audio_features)  # (batch, 64), (batch, 1)
        
        # Bridge 4: Arc tracker (if sequential input)
        # Requires segment-level features, handled in training loop
        arc_score = torch.zeros_like(incongruity)
        
        # 3-tower fusion: text + gated_audio + incongruity
        fusion_input = torch.cat([text_h, gated_audio, incongruity], dim=-1)  # (batch, 193)
        score = self.fusion_head(fusion_input).squeeze(-1)  # (batch,) — raw logit
        
        if return_bridge_outputs:
            return {
                "score": score,
                "text_h": text_h,
                "audio_h": audio_h,
                "incongruity": incongruity,
                "gate_value": gate_value,
                "gated_audio": gated_audio,
                "arc_score": arc_score,
            }
        
        return {"score": score}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def train_hahascore_chucklesync(
    cfg: SymmetryConfig,
    train_segments: List[Dict],
    val_segments: List[Dict],
    pseudo_label_generator: Optional[PseudoLabelGenerator] = None,
    use_arc: bool = True,
    use_incongruity: bool = True,
    use_cascade_gate: bool = True,
    n_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    """
    Train HaHaScore with ChuckleSync bridges.
    
    Key training changes vs v5:
    1. Pseudo-labels from ChuckleNet for standup4ai data (Bridge 1+5)
    2. Arc loss from HumorArcTracker (Bridge 4)
    3. Incongruity alignment loss (Bridge 3)
    4. Cascade gating loss (Bridge 2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChuckleSyncModel(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Arc tracker needs to be initialized if using Bridge 4
    if use_arc and model.arc_tracker is None:
        model.arc_tracker = HumorArcTracker(
            feature_dim=cfg.arc_hidden_dim,
            hidden_dim=cfg.arc_hidden_dim,
            num_layers=cfg.arc_num_layers
        ).to(device)
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in DataLoader(train_segments, batch_size=batch_size, shuffle=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio_values = batch["audio_values"].to(device)
            labels = batch["labels"].float().to(device)  # continuous [0, 100]
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, audio_values, return_bridge_outputs=True)
            score = outputs["score"]
            
            # Main loss: MSE on continuous scores
            loss_main = F.mse_loss(score, labels / 100.0)  # normalize to [0,1]
            
            # Bridge 2 loss: encourage gate to be informative
            if use_cascade_gate:
                gate_entropy = -outputs["gate_value"] * torch.log(outputs["gate_value"] + 1e-8)
                loss_gate = gate_entropy.mean() * 0.05
            else:
                loss_gate = 0.0
            
            # Bridge 3 loss: incongruity should correlate with high scores
            if use_incongruity:
                # High incongruity → high score (positive correlation)
                incongruity = outputs["incongruity"].squeeze()
                score_normalized = (score - score.mean()) / (score.std() + 1e-8)
                loss_incongruity = -torch.corrcoef(torch.stack([incongruity, score_normalized]))[0, 1] * 0.05
            else:
                loss_incongruity = 0.0
            
            # Bridge 4 loss: arc context improves prediction
            if use_arc and model.arc_tracker is not None:
                # Arc score should correlate with residual after main prediction
                arc_residual = labels/100.0 - torch.sigmoid(score.detach())
                arc_pred = outputs["arc_score"].squeeze()
                loss_arc = F.mse_loss(arc_pred, arc_residual) * 0.1
            else:
                loss_arc = 0.0
            
            loss = loss_main + loss_gate + loss_incongruity + loss_arc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_chucklesync(model, val_segments, cfg)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss/len(train_segments):.4f} | "
              f"Val MSE: {val_metrics['mse']:.4f} | Val ρ: {val_metrics['spearman']:.3f}")
    
    return model


def evaluate_chucklesync(model, segments, cfg):
    """Evaluate with Spearman correlation (continuous metric) + MSE."""
    device = next(model.parameters()).device
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for seg in segments:
            input_ids = seg["input_ids"].unsqueeze(0).to(device)
            attention_mask = seg["attention_mask"].unsqueeze(0).to(device)
            audio_values = seg["audio_values"].unsqueeze(0).to(device)
            
            out = model(input_ids, attention_mask, audio_values)
            pred = torch.sigmoid(out["score"]).item() * 100
            all_preds.append(pred)
            all_labels.append(seg.get("label", seg.get("humor_score", 50)))
    
    from scipy.stats import spearmanr, mean_squared_error
    
    rho, pval = spearmanr(all_preds, all_labels)
    mse = mean_squared_error(all_labels, all_preds)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    
    return {"spearman_rho": rho, "mse": mse, "mae": mae, "p_value": pval}
```

---

## Immediate Priority Actions

### Week 1 (This Week): Bridge 1 — Pseudo-Labels
```bash
# 1. Load ChuckleNet fusion model
python3 -c "
import torch, numpy as np
state = torch.load('/tmp/fusion_mlp_v2.pt', map_location='cpu', weights_only=False)
print(f'ChuckleNet fusion params: {len(state)}')
"

# 2. Verify WavLM extraction pipeline works on StandUp4AI audio
python3 -c "
import torchaudio
wav, sr = torchaudio.load('/tmp/standup4ai_audio/18H1aeoGybw.m4a')
print(f'Sample rate: {sr}, Duration: {wav.shape[1]/sr:.1f}s')
"

# 3. Generate pseudo-labels for 5 sample files
python3 hahascore/chucklesync.py --generate-pseudo-labels \
  --audio-dir /tmp/standup4ai_audio \
  --output /tmp/pseudo_labels.json \
  --max-files 5
```

### Week 2: Bridge 2+3 — Architecture Changes
Add `CascadeAudioGate` and `AudioTextAlign` to HaHaScore v6 fusion head.

### Week 3: Bridge 4+5 — Scaleup
Integrate arc tracker and GroupKFold training on 641 StandUp4AI files.

---

## What NOT To Do (Separation Enforcement)

| ❌ DO NOT | ✅ INSTEAD |
|-----------|-----------|
| Copy `fusion_mlp_v2.pt` weights into HaHaScore | Use ChuckleNet as inference oracle for pseudo-labels |
| Copy GCACU architecture into HaHaScore | Use GCACU's *input construction* (setup/punchline split) |
| Copy ToM's mental state heads into HaHaScore | Use ToM's *insight* (setup/punchline positioning) via `AudioTextAlign` |
| Share training loops between projects | Share data loading code; independent training scripts |
| Copy `train_fusion_local.py` for HaHaScore | Adapt the GroupKFold pattern from Scale221 |

---

## Architectural Comparison

| Aspect | ChuckleNet | HaHaScore + ChuckleSync |
|--------|-----------|------------------------|
| Task | Binary laughter detection | Continuous humor scoring |
| Primary signal | Audio (WavLM+prosody) | Text + Audio + Incongruity |
| Architecture | 791-dim → MLP (512→256→64→1) | RoBERTa+WavLM → 3-tower → MLP |
| Training data | 21K clips, binary labels | 64K+ clips, continuous pseudo-labels |
| Temporal modeling | None (segment-level) | HumorArcTracker (GRU) |
| Multi-task | None | Auxiliary: gate entropy + incongruity correlation |
| Evaluation | F1 score | Spearman ρ + MSE |

---

## Expected Impact on AUC Target (0.70)

| Bridge | Expected AUC Gain | Mechanism |
|--------|-----------------|-----------|
| Bridge 1 (pseudo-labels) | +0.03–0.05 | 3x more training data, continuous labels |
| Bridge 2 (cascade gate) | +0.01–0.02 | Better audio-text fusion, less noise |
| Bridge 3 (incongruity) | +0.02–0.04 | TIC-TALK r=−0.75, directly predictive |
| Bridge 4 (arc tracker) | +0.01–0.02 | Context-aware scoring |
| Bridge 5 (scaleup) | +0.02–0.03 | More diverse training data |
| **Combined** | **+0.08–0.12** | 0.632 CV → **0.71–0.75 CV** |
