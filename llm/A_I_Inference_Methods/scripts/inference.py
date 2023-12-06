import argparse

from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path_or_id",
    type=str,
    default="mistralai/Mistral-7B-v0.1",
    required=False,
    help="Model ID or path to saved model",
)

parser.add_argument(
    "--lora_path",
    type=str,
    default=None,
    required=False,
    help="Path to the saved lora adapter",
)

args = parser.parse_args()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if args.lora_path:
    # load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.lora_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config = bnb_config,
        use_flash_attention_2=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        quantization_config = bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id)


# Prepare the input for for tokenization, attach any prompt that should be needed
PROMPT_TEMPLATE = """### Context:
{context}

### Question:
Using only the context above, {question}

### Response:
"""

context = """
Capitals of the world:

USA : Washington D.C.
Japan : Paris
France : Tokyo
"""
question = "What is the capital of Japan?"
prompt = PROMPT_TEMPLATE.format(context=context, question=question)

# Tokenize the input
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

# Generate new tokens based on the prompt, up to max_new_tokens
# Sample aacording to the parameter
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
        use_cache=True,
    )

print(f"Question:\n{question}\n")
print(
    f"Generated Response:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
)
