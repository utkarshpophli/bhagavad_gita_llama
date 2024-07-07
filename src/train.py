import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from .utils import load_config
from .logger import setup_logger

logger = setup_logger(__name__)

def train():
    config = load_config()
    logger.info("Starting training process")

    # Load dataset
    dataset = load_dataset(config['dataset']['name'], split="train")
    logger.info(f"Loaded dataset: {config['dataset']['name']}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['use_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, config['quantization']['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=config['quantization']['use_nested_quant'],
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        r=config['lora']['r'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training arguments
    training_args = TrainingArguments(**config['training'])

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    logger.info("Starting model training")
    trainer.train()
    logger.info("Training completed")

    # Save the model
    trainer.model.save_pretrained(config['model']['new_model'])
    logger.info(f"Model saved as {config['model']['new_model']}")

if __name__ == "__main__":
    train()