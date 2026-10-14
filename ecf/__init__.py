"""ElderlyCodecFake: Detecting Neural Audio Codec Synthesized Elderly Speech Deepfakes"""

__version__ = "1.0.0"
__author__ = "Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Chi-Chun Lee"
__email__ = "girish281003@gmail.com"

from .model import BONSAI
from .fusion import JensenShannonFusion
from .encoders import LanguageBindImageBindEncoder

__all__ = [
    "BONSAI",
    "JensenShannonFusion",
    "LanguageBindImageBindEncoder",
]
