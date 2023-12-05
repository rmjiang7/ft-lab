# Retrieval Augmented Generation

This contains an example of using the Vector DB established in 'I_Vector_DB' to pass into a LLM for generation.

In addition to the notebook, there is a script that can be used to quickly test RAG:

```bash
usage: scripts/rag.py [-h] [--generation_llm GENERATION_LLM] --query QUERY [--top_k TOP_K]

options:
  -h, --help            show this help message and exit
  --generation_llm GENERATION_LLM
                        pretrained model id to use for generation
  --query QUERY         query
  --top_k TOP_K         how many documents to stuff in the rag prompt
```
