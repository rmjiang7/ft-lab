# Finetuing LLMs for RAG

This contains an example of finetuing a base LLM for better use with Retrieval Augmented Generation context prompts.

In addition to the notebook, there is a script to aid in finetuning:

```bash
usage: scripts/finetuning.py [-h] [--model_id MODEL_ID] [--output_dir OUTPUT_DIR] [--num_train_epochs NUM_TRAIN_EPOCHS] [--resume_from_checkpoint]

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   base model to fine tune
  --output_dir OUTPUT_DIR
                        output directory
  --num_train_epochs NUM_TRAIN_EPOCHS
                        number of training epochs
  --resume_from_checkpoint
                        resume training from latest checkpoint
```