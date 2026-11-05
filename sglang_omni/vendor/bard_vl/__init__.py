from transformers import AutoConfig, AutoModelForImageTextToText

from .configuration_bard_vl import BardVLConfig
from .modeling_bard_vl import BardVLForConditionalGeneration

target_model_type = "bard_vl"
AutoConfig.register(target_model_type, BardVLConfig, exist_ok=True)
AutoModelForImageTextToText.register(
    BardVLConfig, BardVLForConditionalGeneration, exist_ok=True
)
