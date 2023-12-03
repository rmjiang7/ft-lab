from langchain.vectorstores.pgvector import PGVector

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import sqlalchemy

# The connection to the database
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver= "psycopg2",
    host = "localhost",
    port = "5432",
    database = "postgres",
    user= "username",
    password="password"
)

# The embedding function that will be used to store into the database
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Creates the database connection to our existing DB
db = PGVector(
    connection_string = CONNECTION_STRING,
    collection_name = "embeddings",
    embedding_function = embedding_function
)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required = True)
    parser.add_argument("--top_k", type=int, default = 2)

    args = parser.parse_args()
    
    # query it
    query = args.query
    docs_with_scores = db.similarity_search_with_score(query, k = 2)
    
    # print results
    for doc, score in docs_with_scores:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)