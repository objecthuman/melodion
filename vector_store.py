from pathlib import Path

import chromadb
from chromadb.config import Settings


chroma_client: chromadb.ClientAPI | None = None


def initialize_client(path: str = "./chroma_data") -> chromadb.ClientAPI:
    """Initialize ChromaDB persistent client."""
    global chroma_client
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client


def get_client() -> chromadb.ClientAPI:
    """Get the global ChromaDB client instance."""
    if chroma_client is None:
        raise RuntimeError("ChromaDB client not initialized. Call initialize_client() first.")
    return chroma_client


def get_or_create_collection(name: str = "music_embeddings"):
    """Get or create a collection in ChromaDB."""
    client = get_client()
    return client.get_or_create_collection(name=name)


def add_embeddings(
    collection_name: str,
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
):
    """Add embeddings to a collection."""
    client = get_client()
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_similar(
    collection_name: str,
    query_embedding: list[float],
    n_results: int = 20,
):
    """Query similar embeddings from a collection."""
    client = get_client()
    collection = client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results
