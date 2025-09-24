"""
Report Generator Node - creates the final research report based on the approved plan
"""
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .base_node import BaseNode
from ..state_manager import AgentState, StateManager
from ..constrained_decoding import JSONDecodingError, create_json_chain
from ..pydentic_models import SectionContent

logger = logging.getLogger(__name__)


class ReportGeneratorNode(BaseNode):
    """Node responsible for generating the final report based on the plan"""
    
    def __init__(self, config: Dict[str, Any], llm_provider, web_search_tool, status_handler=None):
        super().__init__(config, status_handler)
        self.llm_provider = llm_provider
        self.web_search_tool = web_search_tool
        
        # Initialize JSON-constrained parser for section content
        self.section_content_parser = self._create_section_parser()
    
    def _create_section_parser(self):
        """Create the JSON parser for section content"""
        from ..constrained_decoding import create_json_parser
        return create_json_parser(
            SectionContent,
            allow_partial=self.config.get('constrained_decoding', {}).get('allow_partial', True),
            fill_defaults=True
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """Generate the report based on the plan using JSON-constrained decoding"""
        self.update_status("Generating report...")
        
        try:
            report_content = self._generate_full_report(state)
            state["generated_report"] = report_content
            
            self.update_status("Report generation completed.")
            logger.info("Report generation completed with JSON-constrained decoding")
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            logger.error(error_msg)
            StateManager.add_error(state, error_msg)
            state["generated_report"] = "# Research Report\n\nError generating report."
        
        return state
    
    def _generate_full_report(self, state: AgentState) -> str:
        """Generate the complete report content"""
        report_sections = StateManager.get_plan_sections(state)
        report_title = StateManager.get_report_title(state)
        
        report_content = f"## {report_title}\n\n"
        
        for i, section in enumerate(report_sections):
            title = section.get("title", "Untitled Section")
            report_content += f"### {i+1}. {title}\n\n"
            
            self.update_status(f"Generating content for section ({i+1}/{len(report_sections)}): {title}")
            
            # Generate content for this section
            section_content = self._generate_section_content(section, report_title)
            report_content += f"{section_content}\n\n"
        
        return report_content
    
    def _generate_section_content(self, section: Dict[str, Any], report_title: str) -> str:
        """Generate content for a single report section"""
        title = section.get("title", "Untitled Section")
        sub_queries = section.get("sub_queries", [])
        
        # Gather information for all sub-queries
        combined_snippets = self._gather_section_information(sub_queries)
        
        # Try structured generation first
        try:
            content = self._generate_structured_content(combined_snippets, title, report_title)
            return content if content else self._generate_fallback_content(combined_snippets, title, report_title)
        except JSONDecodingError as e:
            logger.warning(f"JSON decoding failed for section '{title}': {e}")
            # Fall back to direct text generation
            return self._generate_fallback_content(combined_snippets, title, report_title)
        except Exception as e:
            logger.warning(f"Structured content generation failed for section '{title}': {e}")
            # Try fallback method
            try:
                return self._generate_fallback_content(combined_snippets, title, report_title)
            except Exception as fallback_error:
                logger.error(f"Both structured and fallback generation failed for '{title}': {fallback_error}")
                return f"Unable to generate content for section: {title}. Please check the LLM configuration and connectivity."
    
    def _gather_section_information(self, sub_queries: list) -> str:
        """Gather information from web search for section sub-queries"""
        combined_snippets = ""
        
        for sub_query in sub_queries:
            try:
                web_results = self.web_search_tool.run(sub_query)
                snippets = [result.get("snippet", "") for result in web_results if result.get("snippet")]
                combined_snippets += "\n".join(snippets) + "\n\n"
            except Exception as e:
                logger.warning(f"Web search failed for sub-query '{sub_query}': {e}")
        
        return combined_snippets
    
    def _generate_structured_content(self, snippets: str, section_heading: str, topic: str) -> str:
        """Generate content using JSON-constrained decoding"""
        prompt_template = """
        Based on the following snippets, generate detailed and coherent section content for the below
        section heading and research topic. The content should be comprehensive and informative. 
        The content should be 200-300 words long, written in natural language, and in paragraph form.

        Snippets: {snippets}
        Section Heading: {section_heading}
        Research Topic: {topic}

        {format_instructions}
        """
        
        # Get the underlying LLM for JSON chain compatibility
        underlying_llm = getattr(self.llm_provider, 'llm', self.llm_provider)
        
        section_chain = create_json_chain(
            llm=underlying_llm,
            pydantic_model=SectionContent,
            allow_partial=True,
            fill_defaults=True
        )
        
        prompt_text = prompt_template.format(
            snippets=snippets,
            section_heading=section_heading,
            topic=topic,
            format_instructions=self.section_content_parser.get_format_instructions()
        )
        
        section_result = section_chain.run(prompt_text)
        
        # Handle cases where content might be empty or missing
        if hasattr(section_result, 'content'):
            content = section_result.content
        else:
            # If result doesn't have content attribute, try to get it as string
            content = str(section_result)
        
        # Ensure we return something meaningful
        return content if content.strip() else f"Can not generated content for section: {section_heading}"
    
    def _generate_fallback_content(self, snippets: str, section_heading: str, topic: str) -> str:
        """Generate content using standard text generation as fallback"""
        prompt_template = """
        Based on the following snippets, generate detailed and coherent section content for the below
        section heading and research topic. The content should be comprehensive and informative. 
        The content should be 200-300 words long, written in natural language, and in paragraph form.

        Snippets: {snippets}
        Section Heading: {section_heading}
        Research Topic: {topic}

        Respond with the generated content for this section. Do not add the section heading.
        """
        
        # Format the prompt manually
        formatted_prompt = prompt_template.format(
            snippets=snippets,
            section_heading=section_heading,
            topic=topic
        )
        
        # Use the LLM provider directly instead of trying to chain it
        try:
            section_content = self.llm_provider.invoke(formatted_prompt)
            logger.info(f"Generated fallback content for section: {section_heading}")
            return section_content
        except Exception as e:
            logger.error(f"Fallback content generation failed: {e}")
            return f"Unable to generate content for section: {section_heading}. Error: {str(e)}"
    
    def _get_output_summary(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced output summary for report generator"""
        base_summary = super()._get_output_summary(state)
        base_summary.update({
            "report_length": len(state.get("generated_report", "")),
            "sections_processed": len(StateManager.get_plan_sections(state))
        })
        return base_summary