from typing import List, Dict

from langchain.vectorstores.pgvector import PGVector
from operator import itemgetter
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableParallel
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host="localhost",
    port="5432",
    database="postgres",
    user="username",
    password="password",
)


def format_docs(docs: List[str]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_pipeline(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1", k: int = 1
):
    model_path_or_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        bnb_4bit_compute_dtype=torch.float16,
        use_flash_attention_2=True,
        load_in_4bit=True,
    )

    embedding_function = SentenceTransformerEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name="embeddings",
        embedding_function=embedding_function,
    )

    retriever = db.as_retriever(search_kwargs={"k": k})
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = PromptTemplate.from_template(
        """
        Answer the question using only this context:
        
        Context: {context}
        
        Question: {question}
        
        Answer: 
        """
    )

    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "sources": lambda input: [
            (doc.page_content, doc.metadata) for doc in input["documents"]
        ],
        "answer": rag_chain_from_docs,
    }
    return rag_chain_with_source


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation_llm",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="pretrained model id to use for generation",
    )
    parser.add_argument("--query", type=str, required=True, help="query")
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="how many documents to stuff in the rag prompt",
    )

    args = parser.parse_args()

    # query it
    rag_chain = build_rag_pipeline(model_id=args.generation_llm, k=args.k)
    res = rag_chain.invoke(args.query)
    print(res["answer"])
