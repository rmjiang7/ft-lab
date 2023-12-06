# Finetuing LLMs for RAG

This contains an example of finetuing a base LLM for better use with Retrieval Augmented Generation context prompts.

In addition to the notebook, there is a script to aid in finetuning:

```bash
usage: scripts/finetuning.py [-h] [--model_id MODEL_ID] [--is_peft] [--output_dir OUTPUT_DIR] [--num_train_epochs NUM_TRAIN_EPOCHS] [--resume_from_checkpoint]

options:
  -h, --help            show this help message and exit
  --model_id MODEL_ID   base model to fine tune
  --is_peft             is this a peft adapter
  --output_dir OUTPUT_DIR
                        output directory
  --num_train_epochs NUM_TRAIN_EPOCHS
                        number of training epochs
  --resume_from_checkpoint
                        resume training from latest checkpoint
```

To train a model from scratch, 

```
python scripts/finetuning.py
```

To continue training a model

```
python scripts/finetuning.py --output_dir path_to_model --is_peft
```