"""
Document Embedder - Processes and embeds retrieved documents
"""
from typing import Dict, Any, List
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """Processes and embeds documents for RAG pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        embedding_model = config.get('vector_store', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = config.get('vector_store', {}).get('chunk_size', 512)
        self.chunk_overlap = config.get('vector_store', {}).get('chunk_overlap', 50)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and embed documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of embedded chunks with metadata
        """
        embedded_chunks = []
        
        for doc in documents:
            try:
                # Extract and clean content
                content = self._extract_content(doc)
                if not content:
                    continue
                
                # Chunk the document
                chunks = self._chunk_document(content, doc)
                
                # Embed chunks
                for chunk in chunks:
                    try:
                        embedding = self.model.encode(chunk['text'])
                        chunk['embedding'] = embedding.tolist()
                        embedded_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to embed chunk: {e}")
                        continue
                
                logger.debug(f"Embedded {len(chunks)} chunks from document: {doc.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to process document {doc.get('title', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def _extract_content(self, doc: Dict[str, Any]) -> str:
        """Extract text content from document"""
        content_parts = []
        
        # Add title
        if doc.get('title'):
            content_parts.append(doc['title'])
        
        # Add abstract/summary
        if doc.get('abstract'):
            content_parts.append(doc['abstract'])
        elif doc.get('summary'):
            content_parts.append(doc['summary'])
        
        # Add main content
        if doc.get('content'):
            content_parts.append(doc['content'])
        
        return '\n\n'.join(content_parts)
    
    def _chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document content with overlapping windows
        
        Args:
            content: Document text content
            metadata: Document metadata
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        words = content.split()
        
        # Simple word-based chunking with overlap
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = ' '.join(words[start:end])
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                break
            
            chunk = {
                'text': chunk_text,
                'chunk_id': chunk_id,
                'start_word': start,
                'end_word': end,
                'source': metadata.get('source', 'unknown'),
                'url': metadata.get('url', ''),
                'title': metadata.get('title', ''),
                'doc_type': metadata.get('doc_type', 'unknown'),
                'publication_date': metadata.get('publication_date', ''),
                'authors': metadata.get('authors', [])
            }
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move start position with overlap
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query for similarity search"""
        try:
            embedding = self.model.encode(query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []
