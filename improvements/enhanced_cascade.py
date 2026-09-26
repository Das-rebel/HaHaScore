#!/usr/bin/env python3
"""
Enhanced Cascade Gate Model Architecture
========================================
Improved version with advanced gating, multi-scale attention,
and hierarchical processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for hierarchical feature extraction."""

    def __init__(self, feature_dim, scales=None):
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.feature_dim = feature_dim
        self.scales = scales
        self.attentions = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in scales
        ])
        self.aggregation = nn.Linear(feature_dim * len(scales), feature_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, feature_dim)
        Returns:
            Tensor of shape (batch, seq_len, feature_dim)
        """
        batch_size, seq_len, _ = x.shape
        scale_features = []

        for scale, attention in zip(self.scales, self.attentions):
            if scale > 1:
                x_down = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
                x_down = F.interpolate(
                    x_down.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                x_down = x
            attended = attention(x_down)
            scale_features.append(attended)

        concatenated = torch.cat(scale_features, dim=-1)
        aggregated = self.aggregation(concatenated)
        return aggregated


class CrossModalAttention(nn.Module):
    """Cross-modal attention between text and audio features."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text, audio, mask=None):
        """
        Args:
            text: Tensor of shape (batch, seq_len, hidden_dim)
            audio: Tensor of shape (batch, seq_len, hidden_dim)
        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size = text.size(0)
        Q = self.query_proj(text)
        K = self.key_proj(audio)
        V = self.value_proj(audio)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.output_proj(context)


class TextConfidenceEstimator(nn.Module):
    """Estimates text confidence with uncertainty quantification."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        params = self.estimator(x)
        mean = torch.sigmoid(params[..., 0:1])
        log_var = params[..., 1:2]
        uncertainty = torch.exp(log_var * 0.5)
        return mean, uncertainty


class DynamicGatingModule(nn.Module):
    """Dynamic gating that combines text confidence with uncertainty estimates."""

    def __init__(self):
        super().__init__()
        self.gating_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, text_conf, text_uncert, audio_uncert):
        gate_input = torch.cat([text_conf, text_uncert, audio_uncert], dim=-1)
        return self.gating_net(gate_input)


class AdvancedBiGRU(nn.Module):
    """Advanced BiGRU with residual connections and layer normalization."""

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * 2
            self.gru_layers.append(
                nn.GRU(layer_input_size, hidden_size, num_layers=1,
                       batch_first=True, bidirectional=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x):
        current = x
        for i in range(self.num_layers):
            gru_out, _ = self.gru_layers[i](current)
            if i > 0 and gru_out.size(-1) == current.size(-1):
                current = gru_out + current
            else:
                current = gru_out
            current = self.layer_norms[i](current)
            current = self.dropouts[i](current)
        return current


class EnhancedCascadeGateFusion(nn.Module):
    """
    Enhanced Cascade Gate Fusion Model with:
    - Multi-scale attention
    - Cross-modal attention
    - Dynamic gating with uncertainty
    - Advanced BiGRU with residuals
    """

    def __init__(self,
                 text_dim=768,
                 audio_dim=791,
                 cross_dim=16,
                 hidden=128,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.cross_dim = cross_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout

        # Project inputs to hidden dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(cross_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Multi-scale attention on hidden representation
        self.text_multi_scale = MultiScaleAttention(hidden)
        self.audio_multi_scale = MultiScaleAttention(hidden)

        # Confidence / uncertainty estimators (input dim is hidden)
        self.text_conf_estimator = TextConfidenceEstimator(hidden)
        self.audio_uncert_estimator = TextConfidenceEstimator(hidden)

        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(hidden)

        # Dynamic gating
        self.dynamic_gating = DynamicGatingModule()

        # Feature fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal modeling
        self.advanced_biGRU = AdvancedBiGRU(hidden, hidden, num_layers, dropout)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, text, audio, cross_modal=None, lengths=None):
        """
        Args:
            text: Tensor of shape (batch, seq_len, text_dim)
            audio: Tensor of shape (batch, seq_len, audio_dim)
            cross_modal: Tensor of shape (batch, seq_len, cross_dim) [optional]
            lengths: Tensor of shape (batch,) [optional]
        Returns:
            scores: Tensor of shape (batch, seq_len, 1)
            text_conf: Tensor of shape (batch, seq_len, 1)
            gate_weight: Tensor of shape (batch, seq_len, 1)
        """
        # Project to hidden
        text_h = self.text_proj(text)
        audio_h = self.audio_proj(audio)

        # Cross-modal projection (default to zeros if not provided)
        if cross_modal is None:
            cross_modal = torch.zeros(audio.shape[0], audio.shape[1],
                                       self.cross_dim, device=audio.device)
        cross_h = self.cross_proj(cross_modal)

        # Multi-scale attention (produces hidden-dim features)
        text_ms = self.text_multi_scale(text_h)
        audio_ms = self.audio_multi_scale(audio_h)

        # Confidence and uncertainty
        text_conf, text_uncert = self.text_conf_estimator(text_ms)
        _, audio_uncert = self.audio_uncert_estimator(audio_ms)

        # Cross-modal attention
        cross_attended = self.cross_modal_attention(text_ms, audio_ms)

        # Dynamic gating
        gate_weight = self.dynamic_gating(text_conf, text_uncert, audio_uncert)

        # Gated audio
        gated_audio = audio_ms * gate_weight

        # Feature fusion
        fused = torch.cat([text_ms, gated_audio, cross_attended, cross_h], dim=-1)
        fused = self.fusion_proj(fused)

        # Temporal modeling
        temporal_output = self.advanced_biGRU(fused)

        # Final prediction
        scores = self.head(temporal_output)
        return scores, text_conf, gate_weight


def create_enhanced_model(text_dim=768, audio_dim=791, cross_dim=16, hidden=128):
    """Factory function to create enhanced model."""
    return EnhancedCascadeGateFusion(
        text_dim=text_dim,
        audio_dim=audio_dim,
        cross_dim=cross_dim,
        hidden=hidden,
        num_layers=2,
        dropout=0.1
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = create_enhanced_model()
    print(f"Model parameters: {count_parameters(model):,}")

    batch_size, seq_len = 2, 20
    text = torch.randn(batch_size, seq_len, 768)
    audio = torch.randn(batch_size, seq_len, 791)
    cross_modal = torch.randn(batch_size, seq_len, 16)

    scores, text_conf, gate_weight = model(text, audio, cross_modal)
    print(f"Scores: {scores.shape}")
    print(f"Text conf: {text_conf.shape}")
    print(f"Gate weight: {gate_weight.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
