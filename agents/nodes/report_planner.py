"""
Report Planning Node - generates structured research plan using LLM
"""
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from .base_node import BaseNode
from ..state_manager import AgentState, StateManager
from ..constrained_decoding import JSONDecodingError, create_json_chain
from ..pydentic_models import ReportPlan

logger = logging.getLogger(__name__)


class ReportPlannerNode(BaseNode):
    """Node responsible for planning the report structure using JSON-constrained decoding"""
    
    def __init__(self, config: Dict[str, Any], llm_provider, web_search_tool, status_handler=None):
        super().__init__(config, status_handler)
        self.llm_provider = llm_provider
        self.web_search_tool = web_search_tool
        
        # Initialize JSON-constrained parser for structured outputs
        self.report_plan_parser = self._create_report_plan_parser()
        
    def _create_report_plan_parser(self):
        """Create the JSON parser for report planning"""
        from ..constrained_decoding import create_json_parser
        return create_json_parser(
            ReportPlan,
            allow_partial=self.config.get('constrained_decoding', {}).get('allow_partial', True),
            fill_defaults=True
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """Plan the report structure using JSON-constrained decoding"""
        self.update_status("Planning report structure...")
        
        query = StateManager.get_query(state)
        
        # Gather context from web search
        context = self._gather_context(query)
        
        # Generate the report plan
        try:
            plan = self._generate_report_plan(query, context)
            StateManager.update_plan(state, plan)
            
            # Set up human review
            StateManager.set_human_review(state, required=True, approved=False)
            
            logger.info(f"Report planning completed. Generated {len(plan.get('report_sections', []))} sections.")
            
        except JSONDecodingError as e:
            error_msg = f"JSON decoding failed for report planning: {str(e)}"
            logger.error(error_msg)
            StateManager.add_error(state, error_msg)
            
            # Provide fallback plan
            fallback_plan = self._create_fallback_plan(query)
            StateManager.update_plan(state, fallback_plan)
            
        except Exception as e:
            error_msg = f"Report planning failed: {str(e)}"
            logger.error(error_msg)
            StateManager.add_error(state, error_msg)
            
            # Provide minimal fallback plan
            fallback_plan = {"report_title": "Research Report", "report_sections": []}
            StateManager.update_plan(state, fallback_plan)
        
        return state
    
    def _gather_context(self, query: str) -> str:
        """Gather context from web search for report planning"""
        web_snippets = []
        
        try:
            self.update_status("Fetching web results for report planning...")
            web_results = self.web_search_tool.run(query)
            
            for result in web_results:
                snippet = result.get("snippet", "")
                if snippet:
                    web_snippets.append(snippet)
                    
        except Exception as e:
            logger.warning(f"Web search for report planning failed: {e}")
        
        return "\n\n".join(web_snippets)
    
    def _generate_report_plan(self, query: str, context: str) -> Dict[str, Any]:
        """Generate the structured report plan using LLM"""
        example_json = (
            '{\n'
            '  "report_sections": [\n'
            '    {"title": "Introduction", "sub_queries": ["What is ...?", "Why is ... important?"]},\n'
            '    {"title": "Key Topic A", "sub_queries": ["How does ...?", "What factors influence ...?"]},\n'
            '    {"title": "Conclusion", "sub_queries": ["What are the key findings about ...?"]}\n'
            '  ],\n'
            '  "report_title": "Concise Title"\n'
            '}'
        )
        
        analysis_prompt = ChatPromptTemplate.from_template(
            """
            You are a research planner. Based on the user's research query, generate a structured outline for a research report in JSON only and no prose.  
            - The report should contain 5-8 sections, beginning with an Introduction and ending with a Conclusion.  
            - For each section, create 2-3 focused sub-queries that are *explicitly and directly* connected to the research query, 
            to guide information gathering.
            - Each sub-query must be complete on its own and related to the research query. 
                Example:
                Topic of research: Tom and Jerry
                Bad example of sub-query: "First theatrical short film release?"
                Good example of sub-query: "When was the first theatrical short film of Tom and Jerry released?"
            - Also find a suitable title for the report. A report title should be clear, concise, and informative, directly related to the user's query and purpose to help readers immediately understand what the report addresses.
            - Use both the provided context and your own knowledge to ensure comprehensive coverage of the topic.

            Context: {context}
            Research Query: {query}

            Example (structure only, use different actual values):
            {example_json}

            {format_instructions}
            """
        )
        
        # Get the underlying LLM for JSON chain compatibility
        underlying_llm = getattr(self.llm_provider, 'llm', self.llm_provider)
        
        json_chain = create_json_chain(
            llm=underlying_llm,
            pydantic_model=ReportPlan,
            allow_partial=True,
            fill_defaults=True
        )
        
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            prompt_text = analysis_prompt.format(
                context=context,
                query=query,
                example_json=example_json,
                format_instructions=self.report_plan_parser.get_format_instructions()
            )
            
            if attempt > 1:
                prompt_text += "\nIf previous output contained schema metadata or invalid structure, correct it. Output ONLY the JSON object."
            
            try:
                result = json_chain.run(prompt_text)
                if hasattr(result, 'model_dump'):
                    return result.model_dump()
                else:
                    return result.dict()
                    
            except JSONDecodingError as e:
                if attempt == max_attempts:
                    raise
                logger.warning(f"Report planning JSON decode failure attempt {attempt}/{max_attempts}: {e}")
                continue
    
    def _create_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Create a fallback plan when JSON decoding fails"""
        return {
            "report_title": f"Research Report: {query[:50]}...",
            "report_sections": [
                {
                    "title": "Introduction", 
                    "sub_queries": [f"What is {query}?", f"Why is {query} important?"]
                },
                {
                    "title": "Conclusion", 
                    "sub_queries": [f"What are the key findings about {query}?"]
                }
            ]
        }
    
    def _get_output_summary(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced output summary for report planner"""
        base_summary = super()._get_output_summary(state)
        plan = state.get("report_plan", {})
        base_summary.update({
            "sections_generated": len(plan.get("report_sections", [])),
            "has_title": bool(plan.get("report_title")),
            "human_review_required": state.get("human_review_required", False)
        })
        return base_summary