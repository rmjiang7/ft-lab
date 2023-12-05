### Inference Apps

This directory contains a few simple starter scripts for how to use your pre-trained or fine-tuned models for inference in different application scenarios.

# Inference of LLMs

In this repo, we give you a few options for using pre-trained for fine-tuned models for inference. 

- [Inference Script](#inference-script)

  This script is a barebones wrapper around loading and calling a model from python.  Use this to build out applications further.  You can also take a look at the `Inference_with_LLMs` notebook for running this interactively.

- [Gradio Frontend](#gradio-frontend)

  This is a commonly used demo app for LLMs that deploys a pretty front-end for showing off inference for a model.

- [Flask API](#flask-api)

  This is a very simple flask api for spinning up an endpoint.  It supports generation via streaming and non-streaming.  It has very low throughput and concurrency due to the expensive nature of LLMs.

## Inference Script

To simple run inference using your trained model, use the `inference.py` script, passing in a finetuned model or a lora adapter.

```
usage: inference.py [-h] [--model_path_or_id MODEL_PATH_OR_ID] [--lora_path LORA_PATH]

options:
  -h, --help            show this help message and exit
  --model_path_or_id MODEL_PATH_OR_ID
                        Model ID or path to saved model
  --lora_path LORA_PATH
                        Path to the saved lora adapter
```

## Gradio Frontend

To bootup a sample application for demos, use the `gradio_app.py` script, passing in a finetuned model or a lora adapter.

```
usage: gradio_app.py [-h] [--model_path_or_id MODEL_PATH_OR_ID] [--lora_path LORA_PATH]

options:
  -h, --help            show this help message and exit
  --model_path_or_id MODEL_PATH_OR_ID
                        Model ID or path to saved model
  --lora_path LORA_PATH
                        Path to the saved lora adapter
```

```
python gradio_app.py
```

Then navigate to `http://localhost:7860` to use the demo app.

## Flask API

To bootup a small flask application for use as an API endpoint, use the `flask_api.py` script, passing in a finetuned model or a lora adapter.

```
usage: flask_api.py [-h] [--model_path_or_id MODEL_PATH_OR_ID] [--lora_path LORA_PATH]

options:
  -h, --help            show this help message and exit
  --model_path_or_id MODEL_PATH_OR_ID
                        Model ID or path to saved model
  --lora_path LORA_PATH
                        Path to the saved lora adapter
```

```
python flask_api.py
```

This api will be hosted at "http://localhost:7861" and have 2 endpoints `/generate` and `/generate/stream` for text generation and streaming.

```
curl -X POST -d '{"prompt" : "This is a test", "parameters" : {"max_new_tokens" : 100, "temperature" : 0.7, "do_sample" : true}}' http://localhost:7861/generate
curl -N -X POST -d '{"prompt" : "This is a test", "parameters" : {"max_new_tokens" : 100, "temperature" : 0.7, "do_sample" : true}}' http://localhost:7861/generate/stream
```
