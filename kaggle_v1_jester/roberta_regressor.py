"""
roberta_regressor.py — RoBERTa regression head (single source-of-truth)
========================================================================
Previously duplicated in 4 files (train_jester_regression.py, v1_jester_m3_recipe.py,
v1_jester_m3_cv_5x3.py, v1_jester_finetuned_kernel.py). Now a single module.

Configurable via constructor kwargs:
- freeze_backbone=True (v1.0 canonical, CPU result)
- use_fp16=True (T4 15GB VRAM)
- use_grad_checkpointing=True (memory savings)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobertaRegressor(nn.Module):
    def __init__(self, model_name='roberta-base', freeze_backbone=False, use_fp16=False,
                 use_grad_checkpointing=False, device='cuda'):
        super().__init__()
        from transformers import RobertaModel
        self.backbone = RobertaModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.use_fp16 = use_fp16
        if device == 'cuda' and use_grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


class MLPHead(nn.Module):
    """CPU-friendly MLP head for cached RoBERTa embeddings (v1 CPU path)."""
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
