import argparse

from flask import Flask, request, Response, stream_with_context
from peft import AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread


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

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)

    # Tokenize the input
    input_ids = tokenizer(
        data["prompt"], return_tensors="pt", truncation=True
    ).input_ids.cuda()

    # Support for streaming of tokens within generate requires
    # generation to run in a separate thread
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=data["parameters"].get("max_new_tokens", 100),
        do_sample=data["parameters"].get("do_sample", True),
        top_p=data["parameters"].get("top_p", 0.9),
        temperature=data["parameters"].get("temperature", 0.7),
        use_cache=True,
    )

    outputs = model.generate(**generation_kwargs)

    return {
        "generated_text": tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0][len(data["prompt"]) :]
    }


@app.route("/generate/stream", methods=["POST"])
def generate_stream():
    data = request.get_json(force=True)

    # Tokenize the input
    input_ids = tokenizer(
        data["prompt"], return_tensors="pt", truncation=True
    ).input_ids.cuda()

    # Support for streaming of tokens within generate requires
    # generation to run in a separate thread
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=data["parameters"].get("max_new_tokens", 100),
        do_sample=data["parameters"].get("do_sample", True),
        top_p=data["parameters"].get("top_p", 0.9),
        temperature=data["parameters"].get("temperature", 0.7),
        use_cache=True,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def f():
        completion = ""
        for r in streamer:
            yield r[len(completion) :]
            completion = r

    return Response(stream_with_context(f()), mimetype="text/event-stream")


app.run(host="0.0.0.0", port=7861)
