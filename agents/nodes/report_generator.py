"""
Report Generator Node - creates the final research report based on the approved plan
"""
import logging
from typing import Dict, Any, List

from .base_node import BaseNode
from ..state_manager import AgentState, StateManager
from ..constrained_decoding import JSONDecodingError, create_json_chain
from ..pydentic_models import SectionContent

logger = logging.getLogger(__name__)


class ReportGeneratorNode(BaseNode):
    """Node responsible for generating the final report based on the plan"""
    
    def __init__(self, config: Dict[str, Any], llm_provider, web_search_tool, rag_tool, status_handler=None):
        super().__init__(config, status_handler)
        self.llm_provider = llm_provider
        self.web_search_tool = web_search_tool
        self.rag_tool = rag_tool

        self.retrieved_docs: List[str] = []
        
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
        
        state["retrieved_docs"] = self.retrieved_docs
        self.retrieved_docs = []

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
        section_heading = section.get("title", "Untitled Section")
        sub_queries = section.get("sub_queries", [])
        
        # Gather information for all sub-queries
        web_results = self._gather_web_results(sub_queries)
        section_context = self._gather_section_context(web_results, section_heading, report_title)

        self.retrieved_docs.append({"section heading": section_heading, "section_context": section_context})
        
        # Try structured generation first
        try:
            content = self._generate_structured_content(section_context, section_heading, report_title)
            return content if content else self._generate_fallback_content(section_context, section_heading, report_title)
        except JSONDecodingError as e:
            logger.warning(f"JSON decoding failed for section '{section_heading}': {e}")
            # Fall back to direct text generation
            return self._generate_fallback_content(section_context, section_heading, report_title)
        except Exception as e:
            logger.warning(f"Structured content generation failed for section '{section_heading}': {e}")
            # Try fallback method
            try:
                return self._generate_fallback_content(section_context, section_heading, report_title)
            except Exception as fallback_error:
                logger.error(f"Both structured and fallback generation failed for '{section_heading}': {fallback_error}")
                return f"Unable to generate content for section: {section_heading}. Please check the LLM configuration and connectivity."
    
    def _gather_web_results(self, sub_queries: list) -> list[str]:
        """Gather information using web search and RAG for section sub-queries"""
        all_documents = []
        
        # Reset RAG tool's vector store for fresh content
        self.rag_tool.reset_vector_store()
        for sub_query in sub_queries:
            try:
                # Use search_and_fetch to get full content from web results
                web_results = self.web_search_tool.search_and_fetch(sub_query, fetch_content=True)
                
                # Extract documents from web results
                for result in web_results:
                    if result.get('documents'):
                        # Add document content to the list
                        all_documents.extend([doc.page_content for doc in result['documents']])
            
            except Exception as e:
                logger.warning(f"Web search and fetch failed for sub-query '{sub_query}': {e}")
        return all_documents
    
    def _gather_section_context(
            self, documents: list[str], 
            section_heading: str, 
            research_topic: str
    ) -> str:
        
        all_documents = documents
        if not all_documents:
            logger.warning("No documents retrieved from web search")
            return ""
        
        try:
            # Create vector store from all gathered documents
            self.rag_tool.create_vector_store(all_documents)
            
            # Use section heading as query to get most relevant documents
            rag_query = f"Find content related to the section heading '{section_heading}' and the research topic '{research_topic}'."
            relevant_docs = self.rag_tool.query_documents(rag_query, top_k=3)

            # Combine relevant documents into context
            section_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            return section_context
            
        except Exception as e:
            logger.error(f"RAG processing failed: {e}")
            # Fallback to using all documents if RAG fails
            return "\n\n".join(all_documents[:3])  # Limit to first 3 documents as fallback
    
    def _generate_structured_content(self, section_context: str, section_heading: str, topic: str) -> str:
        """Generate content using JSON-constrained decoding"""
        prompt_template = """
        Based on the following highly relevant section context retrieved using RAG, generate detailed and coherent 
        section content for the given section heading and research topic. Focus on using the key information and 
        insights from the RAG-retrieved content.

        The content should be:
        - Comprehensive and informative
        - the generated content should be 200-300 words in length
        - Written in natural language with clear paragraph structure
        - Directly related to the section heading and research topic
        - Supported by the provided section-context

        Section Context : {section_context}
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
            section_context=section_context,
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
    
    def _generate_fallback_content(self, section_context: str, section_heading: str, topic: str) -> str:
        """Generate content using standard text generation as fallback"""
        prompt_template = """
        Based on the following highly relevant section context retrieved using RAG, generate detailed and coherent 
        content for the given section heading and research topic. Focus on using the key information and insights 
        from the RAG-retrieved content.

        The content should be:
        - Comprehensive and informative
        - 200-300 words in length
        - Written in natural language with clear paragraph structure
        - Directly related to the section heading and research topic
        - Supported by the provided section-context

        Section Context : {section_context}
        Section Heading: {section_heading}
        Research Topic: {topic}

        Respond with the generated content for this section. Do not add the section heading.
        """
        
        # Format the prompt manually
        formatted_prompt = prompt_template.format(
            section_context=section_context,
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