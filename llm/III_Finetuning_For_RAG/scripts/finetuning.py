from datasets import load_dataset, Dataset
import pandas as pd
from transformers import TrainingArguments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM,
)

import random
import time

from typing import Dict


def load_modified_dataset():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    df = dataset.to_pandas()
    df["keep"] = True

    # Keep entries with correct answer as well
    df = df[
        (df["category"].isin(["closed_qa", "information_extraction", "open_qa"]))
        & df["keep"]
    ]

    return Dataset.from_pandas(
        df[["instruction", "context", "response"]], preserve_index=False
    )


def format_instruction(sample: Dict) -> str:
    """Combine a row to a single str"""
    return f"""### Context:
{sample['context']}

### Question:
Using only the context above, {sample['instruction']}

### Response:
{sample['response']}
"""


def train_model(
    model_id="mistralai/Mistral-7B-v0.1",
    output_dir="mistral-7b-int4-dolly",
    is_peft=False,
    resume_from_checkpoint=False,
    num_train_epochs=1,
):
    dataset = load_modified_dataset()
    # dataset = dataset.select(range(100))

    # Hugging Face Base Model ID
    model_id = model_id
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if is_peft:
        # load base LLM model with PEFT Adapter

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_flash_attention_2=True,
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
        model._mark_only_adapters_as_trainable()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            use_flash_attention_2=True,
        )

        # LoRA config for QLoRA
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "v_proj",
                "down_proj",
                "up_proj",
                "o_proj",
                "q_proj",
                "gate_proj",
                "k_proj",
            ],
        )

        # prepare model for training with low-precision
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # number of training epochs
        per_device_train_batch_size=5,  # batch size per batch
        gradient_accumulation_steps=2,  # effective batch size
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="paged_adamw_32bit",
        logging_steps=1,  # log the training error every 1 steps
        save_strategy="steps",
        save_total_limit=2,
        save_steps=1,  # save a checkpoint every 1 steps
        learning_rate=1e-4,
        ignore_data_skip=True,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        warmup_steps=5,
        lr_scheduler_type="constant",
        disable_tqdm=True,
    )

    # https://huggingface.co/docs/trl/sft_trainer#packing-dataset--constantlengthdataset-
    # max seq length for packing
    max_seq_length = 2048
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        packing=True,
        formatting_func=format_instruction,  # our formatting function which takes a dataset row and maps it to str
        args=args,
    )

    start = time.time()

    # progress bar is fake due to packing
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    end = time.time()
    print(f"{end - start}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="base model to fine tune",
    )

    parser.add_argument(
        "--is_peft",
        action="store_true",
        help="is this a peft adapter",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="mistral-7b-int4-dolly",
        help="output directory",
    )

    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="number of training epochs"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="resume training from latest checkpoint",
    )

    args = parser.parse_args()

    train_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        is_peft=args.is_peft,
        num_train_epochs=args.num_train_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
