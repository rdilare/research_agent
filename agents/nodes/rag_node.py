"""
RAG Node - Retrieval-Augmented Generation pipeline
"""
from typing import Dict, Any, List
import logging
import faiss
import numpy as np
import os

logger = logging.getLogger(__name__)


class RAGNode:
    """Handles RAG pipeline operations including indexing and retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_path = config.get('vector_store', {}).get('index_path', 'embeddings/faiss_index')
        self.index = None
        self.chunk_metadata = []
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        index_file = os.path.join(self.index_path, 'index.faiss')
        metadata_file = os.path.join(self.index_path, 'metadata.json')
        
        if os.path.exists(index_file):
            try:
                self.index = faiss.read_index(index_file)
                # Load metadata if available
                import json
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        self.chunk_metadata = json.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        min_chunks_for_ivf = 40  # FAISS recommends ~40 points per centroid
        default_nlist = 3

        # If you have enough chunks, use IVF-PQ; otherwise, use flat index
        if hasattr(self, 'embedded_chunks') and len(self.embedded_chunks) >= min_chunks_for_ivf:
            nlist = min(default_nlist, len(self.embedded_chunks) // 40)
            nlist = max(1, nlist)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8)
            logger.info(f"Created FAISS IVF-PQ index with nlist={nlist}")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Created FAISS FlatL2 index (fallback for small datasets)")
        self.chunk_metadata = []
    
    def index_chunks(self, embedded_chunks: List[Dict[str, Any]]):
        """Add embedded chunks to the FAISS index"""
        if not embedded_chunks:
            return
        
        try:
            # Extract embeddings and metadata
            embeddings = []
            new_metadata = []
            
            for chunk in embedded_chunks:
                if 'embedding' in chunk:
                    embeddings.append(chunk['embedding'])
                    # Store metadata without embedding for efficiency
                    metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
                    new_metadata.append(metadata)
            
            if not embeddings:
                logger.warning("No embeddings found in chunks")
                return
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Train index if needed (for IVF-based indices)
            if not self.index.is_trained:
                logger.info("Training FAISS index...")
                self.index.train(embeddings_array)
            
            # Add vectors to index
            start_id = self.index.ntotal
            self.index.add(embeddings_array)
            
            # Update metadata
            self.chunk_metadata.extend(new_metadata)
            
            logger.info(f"Added {len(embeddings)} chunks to index. Total: {self.index.ntotal}")
            
            # Save index and metadata
            self._save_index()
            
        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
    
    def retrieve_relevant(self, query: str, embedded_chunks: List[Dict[str, Any]], k: int = 10) -> List[str]:
        """
        Retrieve relevant passages for a query
        
        Args:
            query: User query
            embedded_chunks: List of embedded chunks (will be indexed if needed)
            k: Number of results to retrieve
            
        Returns:
            List of relevant text passages
        """
        try:
            # Index new chunks if provided
            if embedded_chunks:
                self.index_chunks(embedded_chunks)
            
            if self.index.ntotal == 0:
                logger.warning("No vectors in index for retrieval")
                return []
            
            # For this implementation, we'll do a simple similarity search
            # In practice, you'd embed the query using the same model
            from agents.nodes.embedder import Embedder
            embedder = Embedder(self.config)
            query_embedding = embedder.embed_query(query)
            
            if not query_embedding:
                return []
            
            # Search similar vectors
            query_vector = np.array([query_embedding], dtype=np.float32)
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Retrieve corresponding text passages
            relevant_passages = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunk_metadata) and scores[0][i] > 0:  # Valid index and positive score
                    chunk_meta = self.chunk_metadata[idx]
                    passage = chunk_meta.get('text', '')
                    if passage:
                        relevant_passages.append(passage)
            
            logger.info(f"Retrieved {len(relevant_passages)} relevant passages")
            return relevant_passages
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant passages: {e}")
            return []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save FAISS index
            index_file = os.path.join(self.index_path, 'index.faiss')
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            metadata_file = os.path.join(self.index_path, 'metadata.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2)
            
            logger.debug("Saved FAISS index and metadata")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def clear_index(self):
        """Clear the current index"""
        self._create_new_index()
        logger.info("Cleared FAISS index")
