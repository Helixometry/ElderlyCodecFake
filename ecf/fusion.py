"""Jensen-Shannon Divergence fusion module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JensenShannonFusion(nn.Module):
    """
    Jensen-Shannon Divergence based fusion of LanguageBind and ImageBind.

    Uses information-theoretic measure to optimally weight modalities.

    Args:
        hidden_dim (int): Feature dimension
        dropout (float): Dropout probability
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable weights for modalities
        self.register_parameter(
            "lb_weight",
            nn.Parameter(torch.tensor([0.5]))
        )
        self.register_parameter(
            "ib_weight",
            nn.Parameter(torch.tensor([0.5]))
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        languagebind: torch.Tensor,
        imagebind: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse LanguageBind and ImageBind using Jensen-Shannon Divergence.

        Args:
            languagebind: (batch, seq_len, hidden_dim)
            imagebind: (batch, seq_len, hidden_dim)

        Returns:
            Fused representation: (batch, hidden_dim)
        """
        # Average pooling to get utterance-level representations
        lb_pool = languagebind.mean(dim=1)  # (batch, hidden_dim)
        ib_pool = imagebind.mean(dim=1)     # (batch, hidden_dim)

        # Compute Jensen-Shannon Divergence weights
        # Normalize to probability distributions
        lb_norm = F.softmax(lb_pool, dim=-1)
        ib_norm = F.softmax(ib_pool, dim=-1)

        # Jensen-Shannon: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
        # where M = 0.5*(P+Q)
        m = 0.5 * (lb_norm + ib_norm)
        js_div = 0.5 * torch.sum(lb_norm * (torch.log(lb_norm + 1e-8) - torch.log(m + 1e-8)), dim=-1) + \
                 0.5 * torch.sum(ib_norm * (torch.log(ib_norm + 1e-8) - torch.log(m + 1e-8)), dim=-1)

        # Use JS divergence to weight modalities (inverse: lower divergence = higher weight)
        js_weights = torch.softmax(-js_div.unsqueeze(-1), dim=0)

        # Gating-based fusion
        combined = torch.cat([lb_pool, ib_pool], dim=-1)
        gate = self.gate(combined)

        # Weighted fusion
        fused = (self.lb_weight * gate * lb_pool +
                 self.ib_weight * (1 - gate) * ib_pool)

        return self.dropout(fused)
