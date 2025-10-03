"""
Web Search Tool - Provides web search and content fetching capabilities using LangChain tools
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class WebSearchTool:
    """A tool for web searching and content fetching using LangChain components"""
    
    def __init__(self, max_results: int = 5):
        """Initialize the WebSearchTool
        
        Args:
            max_results (int): Maximum number of search results to return
        """
        self.max_results = max_results
        self.search_tool = DuckDuckGoSearchResults(
            max_results=max_results, 
            output_format="list"
        )
        # Configure text splitter for long documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def run(self, query: str) -> List[Dict[str, Any]]:
        """Execute a web search for the given query
        
        Args:
            query (str): Search query string
        
        Returns:
            List[Dict[str, Any]]: List of search results with snippets
        """
        try:
            results = self.search_tool.run(query)
            if isinstance(results, str):
                # If results is a string, try to parse it
                import json
                try:
                    results = json.loads(results)
                except:
                    # Fallback to single result if parsing fails
                    results = [{"snippet": results, "link": None, "title": None}]
            
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]
                
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return []
    
    def fetch_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse content from a webpage using WebBaseLoader
        
        Args:
            url (str): URL of the webpage to fetch
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - documents: List of text chunks for RAG
                - metadata: Source metadata
                - status: Success/error status
        """
        if not url:
            return None
            
        try:
            # Initialize WebBaseLoader
            loader = WebBaseLoader(url)
            
            # Load and process the document
            docs = loader.load()
            if not docs:
                logger.warning(f"No content retrieved from {url}")
                return None
            
            # Split content into chunks
            chunks = self.text_splitter.split_documents(docs)
            
            # Extract metadata from the first document
            metadata = chunks[0].metadata if chunks else {}
            
            # Get all chunks as separate texts
            content_chunks = []
            for chunk in chunks:
                # Combine chunk text with minimal metadata for RAG
                chunk_metadata = {
                    'source': url,
                    'title': metadata.get('title', ''),
                    'chunk_index': len(content_chunks)
                }
                content_chunks.append(Document(
                    page_content=chunk.page_content,
                    metadata=chunk_metadata
                ))
            
            return {
                'title': metadata.get('title', ''),
                'documents': content_chunks,  # List of Document objects for RAG
                'raw_chunks': [chunk.page_content for chunk in chunks],  # Plain text chunks
                'num_chunks': len(chunks),
                'url': url,
                'status': 'success',
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {str(e)}")
            return {
                'url': url,
                'status': 'error',
                'error': str(e)
            }
    
    def search_and_fetch(self, query: str, fetch_content: bool = False) -> List[Dict[str, Any]]:
        """Search and optionally fetch content from search results
        
        Args:
            query (str): Search query string
            fetch_content (bool): Whether to fetch full content from search results
            
        Returns:
            List[Dict[str, Any]]: Search results with optional content and RAG documents
        """
        results = self.run(query)
        
        if fetch_content:
            for result in results:
                if result.get('link'):
                    content_data = self.fetch_content(result['link'])
                    if content_data and content_data['status'] == 'success':
                        # Include both RAG documents and raw text chunks
                        result['documents'] = content_data['documents']
                        result['raw_chunks'] = content_data['raw_chunks']
                        result['num_chunks'] = content_data['num_chunks']
                        result['metadata'] = content_data.get('metadata', {})
                        
        return results
