"""
Tool Orchestrator - Coordinates execution of various research tools
"""
from typing import Dict, Any, List
import logging
from tools.calculator import Calculator
from tools.web_scraper import WebScraper
from tools.data_validator import DataValidator

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """Orchestrates tool execution based on query analysis and context"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools = {
            'calculator': Calculator(config),
            'web_scraper': WebScraper(config),
            'data_validator': DataValidator(config)
        }
    
    def execute_tools(self, query_analysis: Dict[str, Any], context: List[str]) -> Dict[str, Any]:
        """
        Execute relevant tools based on query analysis and context
        
        Args:
            query_analysis: Results from query analyzer
            context: Relevant passages from RAG
            
        Returns:
            Dictionary of tool execution results
        """
        results = {}
        
        try:
            # Determine which tools to use based on query analysis
            required_tools = self._determine_required_tools(query_analysis, context)
            
            logger.info(f"Executing tools: {required_tools}")
            
            for tool_name in required_tools:
                if tool_name in self.tools:
                    try:
                        tool_result = self._execute_tool(tool_name, query_analysis, context)
                        if tool_result:
                            results[tool_name] = tool_result
                            logger.debug(f"Tool {tool_name} executed successfully")
                    except Exception as e:
                        logger.error(f"Tool {tool_name} execution failed: {e}")
                        results[tool_name] = {'error': str(e)}
            
            logger.info(f"Tool execution completed. Results from {len(results)} tools")
            return results
            
        except Exception as e:
            logger.error(f"Tool orchestration failed: {e}")
            return {'error': str(e)}
    
    def _determine_required_tools(self, query_analysis: Dict[str, Any], context: List[str]) -> List[str]:
        """Determine which tools are needed based on query analysis"""
        required_tools = []
        
        intent = query_analysis.get('intent', '').lower()
        domain = query_analysis.get('domain', '').lower()
        entities = [e.lower() for e in query_analysis.get('entities', [])]
        
        # Calculator for financial, mathematical, or statistical analysis
        if any(keyword in intent for keyword in ['calculate', 'analyze', 'compare', 'trend']):
            required_tools.append('calculator')
        
        if any(keyword in domain for keyword in ['finance', 'market', 'economic']):
            required_tools.append('calculator')
        
        # Web scraper for additional data collection
        if 'latest' in intent or 'recent' in intent or 'current' in intent:
            required_tools.append('web_scraper')
        
        # Data validator for quality checks
        if len(context) > 5:  # Only validate if we have substantial data
            required_tools.append('data_validator')
        
        return list(set(required_tools))  # Remove duplicates
    
    def _execute_tool(self, tool_name: str, query_analysis: Dict[str, Any], context: List[str]) -> Any:
        """Execute a specific tool"""
        tool = self.tools[tool_name]
        
        if tool_name == 'calculator':
            return self._execute_calculator(tool, query_analysis, context)
        elif tool_name == 'web_scraper':
            return self._execute_web_scraper(tool, query_analysis, context)
        elif tool_name == 'data_validator':
            return self._execute_data_validator(tool, query_analysis, context)
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return None
    
    def _execute_calculator(self, tool: Calculator, query_analysis: Dict[str, Any], context: List[str]) -> Dict[str, Any]:
        """Execute calculator tool for mathematical analysis"""
        # Extract numerical data from context for analysis
        numerical_data = tool.extract_numbers_from_text(' '.join(context))
        
        results = {}
        if numerical_data:
            results['statistics'] = tool.calculate_statistics(numerical_data)
            results['trends'] = tool.analyze_trends(numerical_data)
        
        return results
    
    
    def _execute_web_scraper(self, tool: WebScraper, query_analysis: Dict[str, Any], context: List[str]) -> Dict[str, Any]:
        """Execute web scraping for additional data"""
        entities = query_analysis.get('entities', [])
        
        results = {}
        for entity in entities[:3]:  # Limit to top 3 entities
            scraped_data = tool.scrape_entity_data(entity)
            if scraped_data:
                results[entity] = scraped_data
        
        return results
    
    def _execute_data_validator(self, tool: DataValidator, query_analysis: Dict[str, Any], context: List[str]) -> Dict[str, Any]:
        """Execute data validation"""
        return tool.validate_context_data(context)
