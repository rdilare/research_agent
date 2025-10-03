"""
RAGTool - A Vector Store based Retrieval-Augmented Generation tool using Chroma.

This module provides a RAG implementation with persistent storage using Chroma vector store
and Ollama embeddings. It handles document storage, retrieval, and management with proper
error handling and logging.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PERSISTENT_DIR = Path(__file__).parent.parent.parent / "data" / "vectorstore"

# Environment settings
os.environ.update({
    'KMP_DUPLICATE_LIB_OK': 'TRUE',  # Handle OpenMP runtime conflict on macOS
})

class RAGTool:
    """A tool for document storage and retrieval using vector embeddings.
    
    This class provides methods for storing, retrieving, and managing documents
    using Chroma vector store with Ollama embeddings. Documents are persisted
    to disk and can be queried for similarity search.
    """

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        
        try:
            # Initialize embeddings model
            self.model_name = model_name
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                base_url=base_url
            )
            
            # Ensure storage directory exists
            PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Initialize Chroma vector store
            self.collection_name = f"rag_collection_{model_name}"
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(PERSISTENT_DIR)
            )
            
            logger.info(
                "Initialized RAGTool with model '%s' and Chroma store at '%s'", 
                model_name, PERSISTENT_DIR
            )
            
        except Exception as e:
            logger.error("Failed to initialize RAGTool: %s", str(e))
            raise

    def create_vector_store(self, documents: List[str]) -> None:
        """Create a new vector store from documents.

        Args:
            documents: List of document strings to index

        Raises:
            Exception: If adding documents fails
        """
        if not documents:
            logger.warning("No documents provided to create vector store")
            return

        try:
            self.vector_store.add_texts(documents)
            self.vector_store.persist()
            logger.info("Created vector store with %d documents", len(documents))
        except Exception as e:
            logger.error("Failed to create vector store: %s", str(e))
            raise

    def query_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most relevant documents for a query.

        Args:
            query: The search query string
            top_k: Maximum number of documents to return

        Returns:
            List of relevant documents with metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            logger.info("Found %d relevant documents for query", len(results))
            return results
        except Exception as e:
            logger.error("Search failed: %s", str(e))
            raise

    def reset_vector_store(self) -> None:
        """Reset vector store to initial empty state.
        
        Raises:
            Exception: If reset operation fails
        """
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(PERSISTENT_DIR)
            )
            logger.info("Successfully reset vector store")
        except Exception as e:
            logger.error("Failed to reset vector store: %s", str(e))
            raise

    def add_documents_to_vector_store(self, documents: List[str]) -> None:
        """Add new documents to existing vector store.

        Args:
            documents: List of document strings to add

        Raises:
            Exception: If adding documents fails
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return

        try:
            self.vector_store.add_texts(documents)
            self.vector_store.persist()
            logger.info("Added %d new documents to vector store", len(documents))
        except Exception as e:
            logger.error("Failed to add documents: %s", str(e))
            raise