### Inference Apps:

## Inference Script

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


## Flask API

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
curl -X POST -d '{"prompt" : "This is a test", "parameters" : {"max_new_tokens" : 100, "temperature" : 0.7, "do_sample" : true}}' http://localhost:8081/generate
curl -N -X POST -d '{"prompt" : "This is a test", "parameters" : {"max_new_tokens" : 100, "temperature" : 0.7, "do_sample" : true}}' http://localhost:8081/generate/stream
```