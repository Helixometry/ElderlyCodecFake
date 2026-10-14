"""Unit tests for BONSAI model."""

import unittest
import torch
from ecf.model import BONSAI


class TestBONSAI(unittest.TestCase):
    """Test BONSAI model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = BONSAI(
            languagebind_dim=768,
            imagebind_dim=1024,
            hidden_dim=512,
        )
        self.batch_size = 4
        self.seq_len = 50

    def test_model_initialization(self):
        """Test model initializes."""
        self.assertIsNotNone(self.model)

    def test_forward_pass(self):
        """Test forward pass."""
        lb = torch.randn(self.batch_size, self.seq_len, 768)
        ib = torch.randn(self.batch_size, self.seq_len, 1024)

        outputs = self.model(lb, ib)

        self.assertIn('logits', outputs)
        self.assertIn('real_prob', outputs)
        self.assertIn('fake_prob', outputs)
        self.assertEqual(outputs['logits'].shape[0], self.batch_size)


if __name__ == '__main__':
    unittest.main()
