from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

class RAGTool:
    def __init__(self):
        """Initialize the RAG tool with an empty vector store."""
        self.vector_store = None

    def create_vector_store(self, documents: List[str]) -> None:
        """
        Create a vector store from raw documents.

        Args:
            documents (List[str]): A list of raw document strings.
        """
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(documents, embeddings)

    def query_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store to retrieve relevant documents.

        Args:
            query (str): The query string.
            top_k (int): The number of top relevant documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of relevant documents with metadata.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Please create it first.")

        results = self.vector_store.similarity_search(query, k=top_k)
        return results

    def reset_vector_store(self) -> None:
        """Reset the vector store to its initial state."""
        self.vector_store = None

    def add_documents_to_vector_store(self, documents: List[str]) -> None:
        """
        Add more documents to the existing vector store.

        Args:
            documents (List[str]): A list of raw document strings to add.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Create a vector store first.")

        embeddings = OpenAIEmbeddings()
        new_vector_store = FAISS.from_texts(documents, embeddings)
        self.vector_store.merge_from(new_vector_store)