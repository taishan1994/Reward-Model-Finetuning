import os

import torch
from accelerate import Accelerator
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

from arguments import ModelArguments


def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def get_peft_config(model_args: ModelArguments):
    if model_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        # task_type="CAUSAL_LM",
        task_type=model_args.lora_task_type,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map():
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None

def get_quantization_config(model_args: ModelArguments):
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config
