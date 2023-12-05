# Document Embeddings and Vector DB

This contains an example of embedding text using and storing them within a Vector DB.  They require `pgvector` to be set up and standing.  This is easiest to do using the `docker-compose.yml`.  

In addition to the notebook, there are 2 scripts that can be used:

For Embedding Documents

```bash
usage: scripts/embed_documents.py [-h] --doc_dir DOC_DIR [--add]

options:
  -h, --help         show this help message and exit
  --doc_dir DOC_DIR  path to document to embed
  --add              add to existing collection
```

For Querying an established Vector DB
```bash
usage: scripts/query_documents.py [-h] --query QUERY [--top_k TOP_K]

options:
  -h, --help     show this help message and exit
  --query QUERY  query
  --top_k TOP_K  how many similar entries to return
```