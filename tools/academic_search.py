"""
Academic Search Tool - Searches academic databases and repositories
"""
from typing import Dict, Any, List
import logging
import requests
import time

logger = logging.getLogger(__name__)


class AcademicSearch:
    """Tool for searching academic databases"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Research-Assistant-Agent/1.0'
        })
    
    def search_arxiv(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search ArXiv for academic papers
        
        Args:
            query: Search query
            analysis: Query analysis results
            
        Returns:
            List of paper metadata and abstracts
        """
        try:
            # ArXiv API endpoint
            base_url = "http://export.arxiv.org/api/query"
            max_results = self.config.get('data_sources', {}).get('arxiv', {}).get('max_results', 10)
            
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (simplified - would use proper XML parsing)
            papers = self._parse_arxiv_response(response.text)
            
            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for academic papers (DISABLED)
        
        Args:
            query: Search query
            analysis: Query analysis results
            
        Returns:
            Empty list (feature disabled)
        """
        logger.info("Semantic Scholar search is disabled")
        return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML response (simplified implementation)"""
        papers = []
        
        # This is a simplified parser - in practice, use proper XML parsing
        import re
        
        # Extract entries (very basic regex parsing)
        entry_pattern = r'<entry>(.*?)</entry>'
        entries = re.findall(entry_pattern, xml_content, re.DOTALL)
        
        for entry in entries:
            try:
                paper = {
                    'source': 'arxiv',
                    'doc_type': 'academic_paper'
                }
                
                # Extract title
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                if title_match:
                    paper['title'] = title_match.group(1).strip()
                
                # Extract abstract
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                if summary_match:
                    paper['abstract'] = summary_match.group(1).strip()
                    paper['content'] = paper['abstract']  # Use abstract as main content
                
                # Extract authors
                author_pattern = r'<name>(.*?)</name>'
                authors = re.findall(author_pattern, entry)
                paper['authors'] = authors
                
                # Extract URL
                id_match = re.search(r'<id>(.*?)</id>', entry)
                if id_match:
                    paper['url'] = id_match.group(1).strip()
                
                # Extract publication date
                published_match = re.search(r'<published>(.*?)</published>', entry)
                if published_match:
                    paper['publication_date'] = published_match.group(1).strip()
                
                papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Failed to parse ArXiv entry: {e}")
                continue
        
        return papers
    
    def _format_semantic_scholar_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """Format Semantic Scholar API results"""
        papers = []
        
        for result in results:
            try:
                paper = {
                    'source': 'semantic_scholar',
                    'doc_type': 'academic_paper',
                    'title': result.get('title', ''),
                    'abstract': result.get('abstract', ''),
                    'content': result.get('abstract', ''),  # Use abstract as main content
                    'authors': [author.get('name', '') for author in result.get('authors', [])],
                    'publication_date': str(result.get('year', '')),
                    'citation_count': result.get('citationCount', 0),
                    'url': result.get('url', ''),
                    'paper_id': result.get('paperId', '')
                }
                
                papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Failed to format Semantic Scholar result: {e}")
                continue
        
        return papers
    
    def search_pubmed(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search PubMed for medical/biological papers (placeholder implementation)"""
        # This would implement PubMed API integration
        logger.info("PubMed search not yet implemented")
        return []
    
    def search_google_scholar(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Google Scholar (placeholder - would require scraping or unofficial API)"""
        # This would implement Google Scholar search
        logger.info("Google Scholar search not yet implemented")
        return []
