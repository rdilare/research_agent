"""
Query Analyzer - Determines research domain and source selection strategy
"""
from typing import Dict, Any, List
import logging
from llm.llama_interface import LlamaInterface

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes user queries to determine research domain and source selection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = LlamaInterface(config.get('llm', {}))
    
    def analyze(self, user_prompt: str) -> Dict[str, Any]:
        """
        Analyze user query to extract domain, entities, intent, and source recommendations
        
        Args:
            user_prompt: The user's research question
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            analysis_prompt = self._build_analysis_prompt(user_prompt)
            response = self.llm.generate(analysis_prompt)
            
            # Parse LLM response into structured format
            analysis = self._parse_analysis_response(response)
            
            # Add source recommendations based on domain
            analysis['source_recommendations'] = self._recommend_sources(analysis)
            
            logger.info(f"Query analysis completed for domain: {analysis.get('domain')}")
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return default analysis
            return {
                'domain': 'general',
                'entities': [],
                'intent': 'research',
                'topics': [user_prompt],
                'source_recommendations': ['duckduckgo']
            }
    
    def _build_analysis_prompt(self, user_prompt: str) -> str:
        """Build prompt for LLM analysis"""
        return f"""
        Analyze the following research query and extract:
        1. Domain (academic, market, technology, general, etc.)
        2. Key entities and concepts
        3. Research intent (overview, comparison, analysis, etc.)
        4. Main topics/themes
        
        Query: "{user_prompt}"
        
        Respond in this format:
        Domain: [domain]
        Entities: [entity1, entity2, ...]
        Intent: [intent]
        Topics: [topic1, topic2, ...]
        """
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis"""
        analysis = {
            'domain': 'general',
            'entities': [],
            'intent': 'research',
            'topics': []
        }
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith('Domain:'):
                    analysis['domain'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Entities:'):
                    entities_str = line.split(':', 1)[1].strip()
                    analysis['entities'] = [e.strip() for e in entities_str.split(',') if e.strip()]
                elif line.startswith('Intent:'):
                    analysis['intent'] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Topics:'):
                    topics_str = line.split(':', 1)[1].strip()
                    analysis['topics'] = [t.strip() for t in topics_str.split(',') if t.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse analysis response: {e}")
        
        return analysis
    
    def _recommend_sources(self, analysis: Dict[str, Any]) -> List[str]:
        """Recommend data sources based on analysis"""
        domain = analysis.get('domain', 'general')
        sources = []
        
        # Academic domain
        if 'academic' in domain or 'research' in domain or 'scientific' in domain:
            if self.config.get('data_sources', {}).get('arxiv', {}).get('enabled', False):
                sources.append('arxiv')
        
        # Always include general web search as fallback
        if self.config.get('data_sources', {}).get('duckduckgo', {}).get('enabled', True):
            sources.append('duckduckgo')
        
        return sources if sources else ['duckduckgo']
