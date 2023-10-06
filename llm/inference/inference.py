import argparse

from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--model_path_or_id", 
                    type=str, 
                    default = "NousResearch/Llama-2-7b-hf", 
                    required = False,
                    help = "Model ID or path to saved model")

parser.add_argument("--lora_path", 
                    type=str, 
                    default = None, 
                    required = False,
                    help = "Path to the saved lora adapter")

args = parser.parse_args()

if args.lora_path:
    # load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.lora_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenier = AutoTokenizer.from_pretrained(args.lora_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id)

# Prepare the input for for tokenization, attach any prompt that should be needed
PROMPT_TEMPLATE = """### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{instruction}

### Response:
"""
instruction = "Tell me all of the moon phases."

# Tokenize the input
input_ids = tokenizer(
    PROMPT_TEMPLATE.format(instruction=instruction), 
    return_tensors="pt", 
    truncation=True).input_ids.cuda()

# Generate new tokens based on the prompt, up to max_new_tokens
# Sample aacording to the parameter
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        do_sample=True, 
        top_p=0.9,
        temperature=0.9,
        use_cache=True
    )

print(f"Prompt:\n{instruction}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(instruction):]}")