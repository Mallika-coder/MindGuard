"""
Build FAISS Vector Store from mental health knowledge base.

This script creates the vector embeddings and saves the FAISS index
for the RAG pipeline to use during inference.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from backend.app.rag_pipeline import KNOWLEDGE_BASE

VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")


def build_and_save():
    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = [
        Document(page_content=item["content"], metadata={"topic": item["topic"]})
        for item in KNOWLEDGE_BASE
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")

    print("Building FAISS index...")
    vector_store = FAISS.from_documents(splits, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_DIR)
    print(f"Vector store saved to: {VECTOR_STORE_DIR}")

    print("\nTesting retrieval...")
    results = vector_store.similarity_search("I feel anxious and can't sleep", k=3)
    for i, doc in enumerate(results):
        print(f"\n  Result {i+1} [{doc.metadata['topic']}]:")
        print(f"  {doc.page_content[:100]}...")


if __name__ == "__main__":
    build_and_save()
