from datasets import load_dataset, Dataset
import pandas as pd
from transformers import TrainingArguments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import random
import time

from typing import Dict


def load_modified_dataset():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Randomly mix responses where context and question mismatch
    df = dataset.to_pandas()
    df["keep"] = True
    oqa = df[df["category"] == "open_qa"]
    cqa = df[df["category"] == "closed_qa"]
    oi = list(range(len(oqa)))[:1000]
    ci = list(range(len(cqa)))[:1000]
    oqa = oqa.iloc[oi]
    cqa = cqa.iloc[ci]
    oqa["context"] = cqa["context"].values
    oqa["response"] = "There is no information about this in the context."

    # Add situation when no context is provided
    def no_context_response(x):
        if x["category"] == "open_qa":
            x["response"] = f"I don't know the answer to {x['instruction']}"
            if random.uniform(0, 1) < 0.2:
                x["keep"] = False
        return x

    df = df.apply(no_context_response, axis=1)

    # Keep entries with correct answer as well
    df = df[
        (df["category"].isin(["closed_qa", "information_extraction", "open_qa"]))
        & df["keep"]
    ]

    return Dataset.from_pandas(
        pd.concat([df, oqa])[["instruction", "context", "response"]],
        preserve_index=False,
    )


def format_instruction(sample: Dict) -> str:
    """Combine a row to a single str"""
    return f"""### System:
You are an information extraction system.  Use only the Context provide below to answer the Question.

### Context:
{sample['context']}

### Question:
{sample['instruction']}

### Response:
{sample['response']}
"""


def train_model(
    model_id="mistralai/Mistral-7B-v0.1",
    output_dir="mistral-7b-int4-dolly",
    resume_from_checkpoint=False,
    num_train_epochs=1,
):
    dataset = load_modified_dataset()
    #dataset = dataset.select(range(100))

    # Hugging Face Base Model ID
    model_id = model_id
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        use_flash_attention_2=True,
        use_cache=False, 
        device_map="auto")
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"

    # LoRA config for QLoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
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

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # number of training epochs
        per_device_train_batch_size=6,  # batch size per batch
        gradient_accumulation_steps=2,  # effective batch size
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,  # log the training error every 10 steps
        save_strategy="steps",
        save_total_limit=2,
        save_steps=1,  # save a checkpoint every 1 steps
        learning_rate=5e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=1.0,
        warmup_steps=5,
        lr_scheduler_type="constant",
        disable_tqdm=True
    )

    # https://huggingface.co/docs/trl/sft_trainer#packing-dataset--constantlengthdataset-
    # max seq length for packing
    max_seq_length = 2048
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
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
        num_train_epochs=args.num_train_epochs,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
