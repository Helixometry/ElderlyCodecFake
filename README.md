# ElderlyCodecFake: Detecting Neural Audio Codec Synthesized Elderly Speech Deepfakes

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202026-blue)](https://www.isca-archive.org/interspeech_2026/phukan26_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)

Detecting neural audio codec synthesized deepfakes in elderly speech using multimodal foundation models

</div>

## Overview

**ElderlyCodecFake (ECF)** addresses a critical vulnerability: existing deepfake detectors fail to generalize to elderly speech. We introduce the **Elderly CodecFake Detection (ECFD) task** and the **ECF dataset** covering English and Chinese elderly speech.

### Key Problem

Current audio codec deepfake detectors trained on existing benchmarks **generalize poorly to elderly speech** because:
- Elderly voices have different acoustic characteristics
- Age-related speech patterns (slower tempo, lower pitch variability)
- Existing datasets lack elderly speaker representation

### Key Solution

**BONSAI**: A multimodal fusion framework that:
1. Leverages **LanguageBind** (cross-modal speech-text pretraining)
2. Leverages **ImageBind** (unified multimodal embedding space)
3. Uses **Jensen-Shannon Divergence** for optimal fusion
4. Achieves **1.66% average EER** on elderly speech

## Paper

**Bridging the Age Gap: Towards Detecting Neural Audio Codec Synthesized Elderly Speech Deepfake**

- **Authors**: Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Chi-Chun Lee
- **Conference**: Interspeech 2026, pp. 5435–5440
- **Paper Link**: [ISCA Archive](https://www.isca-archive.org/interspeech_2026/phukan26_interspeech.html)

## Dataset & Task

### Elderly-CodecFake (ECF) Dataset

- **Languages**: English + Chinese
- **Focus**: Elderly speakers
- **Synthesis Method**: Neural audio codecs
- **Task**: Binary classification (Real/Fake)

### Coverage

- Multiple audio codec architectures
- Real elderly speech samples
- Natural and synthetic speech variations

## BONSAI Framework

### Architecture

```
Input Audio
    ↓
┌─────────────────────────────────┐
│   LanguageBind Encoder          │
│   (Cross-modal speech-text)     │
└──────────┬──────────────────────┘
           ↓
    Speech-Text Embeddings
           ↓
┌─────────────────────────────────┐
│   ImageBind Encoder             │
│   (Unified multimodal space)    │
└──────────┬──────────────────────┘
           ↓
    Multimodal Embeddings
           ↓
┌─────────────────────────────────┐
│   Jensen-Shannon Fusion         │
│   (Optimal modality weighting)  │
└──────────┬──────────────────────┘
           ↓
    Fused Representation
           ↓
┌─────────────────────────────────┐
│   Classification Head           │
│   (Real vs Fake)                │
└─────────────────────────────────┘
```

### Components

1. **LanguageBind**: Speech-text cross-modal pretraining
2. **ImageBind**: Unified multimodal embedding space
3. **Jensen-Shannon Divergence**: Information-theoretic fusion

## Quick Start

### Installation

```bash
git clone https://github.com/Helixometry/ElderlyCodecFake.git
cd ElderlyCodecFake

pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config configs/config_bonsai.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/bonsai_best.pt --test-data data/ecf_test
```

### Inference

```bash
python scripts/inference.py --audio elderly_speech.wav --checkpoint models/bonsai_best.pt
```

## Results

### Performance on ECF Dataset

| Model | English EER | Chinese EER | Average EER |
|-------|-------------|-------------|------------|
| LanguageBind Only | 3.2% | 4.1% | 3.65% |
| ImageBind Only | 2.8% | 3.9% | 3.35% |
| **BONSAI (LB+IB)** | **1.5%** | **1.8%** | **1.66%** |

### Comparison with Baselines

| Baseline | EER |
|----------|-----|
| Traditional ASVspoof Models | 8.2% |
| Standard Codec Detectors | 6.5% |
| Single Encoders | 3.3% |
| **BONSAI** | **1.66%** |

## Repository Structure

```
ElderlyCodecFake/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├──
├── ecf/                         # Main package
│   ├── __init__.py
│   ├── model.py                 # BONSAI model
│   ├── fusion.py                # Jensen-Shannon fusion
│   ├── encoders.py              # LanguageBind/ImageBind wrappers
│   ├── utils.py
│   └── metrics.py
│
├── data/                        # Data handling
│   ├── __init__.py
│   ├── loaders.py               # ECF dataset loaders
│   ├── preprocessor.py          # Audio preprocessing
│   └── augmentation.py          # Data augmentation
│
├── scripts/                     # Training/evaluation
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── __init__.py
│
├── configs/                     # Configuration files
│   ├── config_bonsai.yaml
│   └── config_baseline.yaml
│
├── tests/                       # Unit tests
│   ├── test_model.py
│   ├── test_fusion.py
│   └── __init__.py
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   └── API.md
│
├── audio/                       # Sample audio files
├── fig/                         # Figures and diagrams
└── results/                     # Results and checkpoints
```

## Key Features

✨ **Multimodal Fusion** - LanguageBind + ImageBind  
🌍 **Cross-Lingual** - English and Chinese support  
👴 **Elderly-Focused** - Specialized for elderly speech  
📊 **SOTA Performance** - 1.66% average EER  
🔬 **Foundation Models** - Leverages pretrained multimodal models  
🎯 **Jensen-Shannon Fusion** - Information-theoretic combination  

## Citation

```bibtex
@inproceedings{phukan2026elderlycodecfake,
  title={Bridging the Age Gap: Towards Detecting Neural Audio Codec Synthesized Elderly Speech Deepfake},
  author={Phukan, Orchid Chetia and Girish and Akhtar, Mohd Mujtaba and Lee, Chi-Chun},
  booktitle={Proceedings of Interspeech 2026},
  pages={5435--5440},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Orchid Chetia Phukan**
- **Girish** - girish281003@gmail.com
- **Mohd Mujtaba Akhtar**
- **Chi-Chun Lee**

## Acknowledgments

- Interspeech 2026 organizers
- LanguageBind and ImageBind model developers
- Elderly speech research community
- Audio deepfake detection community

---

<div align="center">

**Built with ❤️ for protecting elderly voices from deepfakes**

</div>