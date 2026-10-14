# BONSAI Architecture for Elderly Codec Deepfake Detection

## Overview

BONSAI (Bridging the Age Gap with Optimized Neural Synthesis Aging Inference) detects neural audio codec synthesized deepfakes in elderly speech using multimodal foundation models.

## Core Problem

Existing deepfake detectors fail on elderly speech because:
- Age-related acoustic variations (slower tempo, lower pitch variability)
- Different prosodic patterns in elderly speakers
- Existing benchmarks lack elderly speech representation

## Solution: BONSAI

Synergizes two multimodal foundation models:

### 1. LanguageBind (LB)
- Cross-modal speech-text pretraining
- Captures semantic and linguistic information
- 768-dimensional embeddings

### 2. ImageBind (IB)
- Unified multimodal embedding space
- Captures diverse modality relationships
- 1024-dimensional embeddings

### 3. Jensen-Shannon Divergence Fusion
- Information-theoretic measure
- Optimal modality weighting
- Adaptive to elderly speech characteristics

## Architecture Flow

```
Audio Input
    ↓
┌─────────────────────────────┐
│ LanguageBind Encoder        │
│ (Speech-Text)              │
│ Output: 768-dim            │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ ImageBind Encoder           │
│ (Multimodal)               │
│ Output: 1024-dim           │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ Jensen-Shannon Fusion       │
│ (Information-theoretic)     │
│ Output: 512-dim            │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ Classification Head         │
│ (Real/Fake)                │
└─────────────────────────────┘
```

## Jensen-Shannon Divergence

Mathematical foundation:
```
JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
where M = 0.5 * (P + Q)
```

Benefits:
- Symmetric divergence measure
- Bounded between 0 and 1
- Natural weight computation from modal distributions
- More robust than simple concatenation

## Performance

### ECF Dataset Results

| Component | English EER | Chinese EER | Avg EER |
|-----------|-------------|-------------|---------|
| LB Only | 3.2% | 4.1% | 3.65% |
| IB Only | 2.8% | 3.9% | 3.35% |
| **BONSAI (LB+IB)** | **1.5%** | **1.8%** | **1.66%** |

### Improvements Over Baselines

- +6.54% over ASVspoof models
- +4.84% over standard codec detectors
- +1.69% over single encoders

## Key Innovation

**Multimodal Fusion for Elderly Speech**: By combining:
- Linguistic patterns (LanguageBind)
- Multimodal relationships (ImageBind)
- Information-theoretic fusion (JS Divergence)

BONSAI achieves robust detection across age groups.

## Cross-Lingual Capability

- Trained on multilingual data (English + Chinese)
- Generalizes across languages
- Age-invariant representations

## Configuration

Paper hyperparameters:
- LanguageBind encoder: frozen, 768-dim
- ImageBind encoder: frozen, 1024-dim
- Fusion hidden: 512-dim
- Batch size: 16
- Learning rate: 1e-4 (AdamW)
- Training epochs: 50
- EER metric: primary evaluation
