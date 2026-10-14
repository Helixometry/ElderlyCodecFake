"""BONSAI model for elderly codec deepfake detection."""

import torch
import torch.nn as nn
from typing import Dict, Optional


class BONSAI(nn.Module):
    """
    BONSAI: Multimodal fusion framework for elderly codec deepfake detection.

    Fuses LanguageBind and ImageBind embeddings using Jensen-Shannon Divergence.

    Args:
        languagebind_dim (int): LanguageBind embedding dimension
        imagebind_dim (int): ImageBind embedding dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        languagebind_dim: int = 768,
        imagebind_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.languagebind_dim = languagebind_dim
        self.imagebind_dim = imagebind_dim
        self.hidden_dim = hidden_dim

        # Projection layers
        self.lb_proj = nn.Linear(languagebind_dim, hidden_dim)
        self.ib_proj = nn.Linear(imagebind_dim, hidden_dim)

        # Jensen-Shannon fusion
        from .fusion import JensenShannonFusion
        self.fusion = JensenShannonFusion(hidden_dim, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary: Real/Fake
        )

    def forward(
        self,
        languagebind_features: torch.Tensor,
        imagebind_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BONSAI.

        Args:
            languagebind_features: LanguageBind embeddings (batch, seq_len, languagebind_dim)
            imagebind_features: ImageBind embeddings (batch, seq_len, imagebind_dim)

        Returns:
            Dictionary with logits and probabilities
        """
        # Project to common space
        lb_proj = self.lb_proj(languagebind_features)
        ib_proj = self.ib_proj(imagebind_features)

        # Jensen-Shannon fusion
        fused = self.fusion(lb_proj, ib_proj)

        # Classification
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "real_prob": probs[:, 0],
            "fake_prob": probs[:, 1],
            "features": fused,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "BONSAI":
        """Load pretrained BONSAI model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """Save BONSAI model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "languagebind_dim": self.languagebind_dim,
                "imagebind_dim": self.imagebind_dim,
                "hidden_dim": self.hidden_dim,
            },
        }
        torch.save(checkpoint, checkpoint_path)
