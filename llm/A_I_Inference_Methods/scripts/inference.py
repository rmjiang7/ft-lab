import argparse

from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--model_path_or_id", 
                    type=str, 
                    default = "mistralai/Mistral-7B-v0.1", 
                    required = False,
                    help = "Model ID or path to saved model")

parser.add_argument("--lora_path", 
                    type=str, 
                    default = None, 
                    required = False,
                    help = "Path to the saved lora adapter")

args = parser.parse_args()

if lora_path:
    # load base LLM model with PEFT Adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_id)

# Prepare the input for for tokenization, attach any prompt that should be needed
PROMPT_TEMPLATE = """
Use the following context to answer the question.

### Context:

### Question:
{question}

### Response:
"""
question = "Tell me all of the moon phases."
prompt = PROMPT_TEMPLATE.format(question=question)

# Tokenize the input
input_ids = tokenizer(
    prompt,
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

print(f"Question:\n{question}\n")
print(f"Generated Response:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")