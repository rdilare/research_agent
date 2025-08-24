"""
Unified Retriever - Dynamically selects and retrieves from multiple data sources
"""
from typing import Dict, Any, List
import logging
from tools.academic_search import AcademicSearch
from tools.market_search import MarketSearch
from tools.web_scraper import WebScraper

logger = logging.getLogger(__name__)


class UnifiedRetriever:
    """Single retriever that dynamically selects data sources based on query analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.academic_search = AcademicSearch(config)
        self.market_search = MarketSearch(config)
        self.web_scraper = WebScraper(config)
    
    def retrieve(self, user_prompt: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve data from sources based on query analysis
        
        Args:
            user_prompt: Original user query
            query_analysis: Analysis results from QueryAnalyzer
            
        Returns:
            List of retrieved documents with metadata
        """
        all_results = []
        source_recommendations = query_analysis.get('source_recommendations', ['duckduckgo'])
        
        logger.info(f"Retrieving from sources: {source_recommendations}")
        
        for source in source_recommendations:
            try:
                results = self._retrieve_from_source(source, user_prompt, query_analysis)
                all_results.extend(results)
                logger.info(f"Retrieved {len(results)} documents from {source}")
            except Exception as e:
                logger.error(f"Failed to retrieve from {source}: {e}")
                continue
        
        # Deduplicate and rank results
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(deduplicated_results, query_analysis)
        
        logger.info(f"Total unique results: {len(ranked_results)}")
        return ranked_results
    
    def _retrieve_from_source(self, source: str, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve from a specific source"""
        
        if source == 'arxiv':
            return self.academic_search.search_arxiv(query, analysis)
        elif source == 'duckduckgo':
            return self.market_search.search_duckduckgo(query, analysis)
        else:
            logger.warning(f"Unknown source: {source}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL or title similarity"""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            url = result.get('url', '')
            title = result.get('title', '').lower()
            
            # Skip if URL already seen
            if url and url in seen_urls:
                continue
            
            # Skip if very similar title already seen
            if title and any(self._similarity_check(title, seen_title) > 0.8 for seen_title in seen_titles):
                continue
            
            seen_urls.add(url)
            seen_titles.add(title)
            unique_results.append(result)
        
        return unique_results
    
    def _similarity_check(self, text1: str, text2: str) -> float:
        """Simple similarity check based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _rank_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank results based on relevance to query analysis"""
        
        # Simple ranking based on title/content relevance to topics
        topics = analysis.get('topics', [])
        entities = analysis.get('entities', [])
        search_terms = topics + entities
        
        for result in results:
            score = 0
            title = result.get('title', '').lower()
            content = result.get('content', '').lower()
            
            # Score based on search term matches
            for term in search_terms:
                term_lower = term.lower()
                score += title.count(term_lower) * 2  # Title matches weighted higher
                score += content.count(term_lower)
            
            # Boost academic sources for academic queries
            if analysis.get('domain') == 'academic' and result.get('source') in ['arxiv']:
                score *= 1.5
            
            result['relevance_score'] = score
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
