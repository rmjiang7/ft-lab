# Large Language Model Practice Experiments

## Installation

Run the following commands to install the necessary libraries.
```bash
# Run this if you haven't already 
# docker compose up -d 

pip install -r requirements.txt
bash install_flash_attention2.sh
```

## Description

The `llm` project consists of examples of RAG and Finetuning.  It is divided into the following sections:

1. Appendix-I A_I_Inference_Methods

    This is a reference example of ways to run and generate text from open source LLMs.  This includes calling the model from code, using `gradio` to deploy a quick chat interface, and a `flask` api for development of service based applications. 
  
2. I_Vector_DB
   
    This section explores text embeddings, generating embeddings from open source models, and storing the embeddings into a Vector DB for Vector search using `pgvector` and `langchain`.

3. II_RAG_Prompt_Engineering

    This section looks at how to use the Vector DB constructed in `I_Vector_DB` with pre-trained open source LLMs to build simple RAG based systems.  Requires execution of `I_Vector_DB` to use correctly.

2. III_Finetuning_For_RAG

    This section demonstrates how to fine tune a base LLM for use within a RAG application, using what we learned in `I_Vector_DB` and `II_RAG_Prompt_Engineering` to customize a base LLM to better utilize context.
