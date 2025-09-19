"""
LangGraph Agent Workflow Orchestration using Standard LangChain/LangGraph Components
"""
from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama  # Use community Ollama instead
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Standard LangGraph state definition using TypedDict"""
    messages: Annotated[list, add_messages]
    original_query: str
    query_analysis: Dict[str, Any]
    raw_documents: List[Dict[str, Any]]
    retrieved_docs: List[str]
    generated_text: str
    markdown_report: str
    timestamp: str
    report_plan: Dict[str, Any]
    generated_report: str
    errors: List[str]
    
    # Human-in-the-loop fields
    human_review_required: bool
    human_approved: bool
    human_feedback: str
    modified_plan: Dict[str, Any]
    
    # Execution tracking
    step_logs: List[Dict[str, Any]]
    execution_times: Dict[str, float]


class ResearchAgentGraph:
    """Standard LangGraph-based research agent workflow"""
    
    def __init__(self, config: Dict[str, Any], status_handler=None):
        self.config = config
        self.status_handler = status_handler
        self.progress_max_count = 8  # Default max count for status bar
        self.progress_counter = 0

        # Initialize standard LangChain components
        self.llm = Ollama(
            model=config.get('llm', {}).get('model', 'llama3.2'),
            base_url=config.get('llm', {}).get('base_url', 'http://localhost:11434'),
            temperature=config.get('llm', {}).get('temperature', 0.7)
        )
        
        # Standard LangChain retrievers and tools
        self.web_search_tool = DuckDuckGoSearchResults(max_results=5, output_format="list")
        
        # Initialize memory checkpointer for thread-based execution
        self.checkpointer = MemorySaver()
        
        # Build the workflow graph
        self.graph = self._build_graph()

        print("="*20, " Langchain Graph ", "="*20)
        print(self.graph.get_graph().draw_mermaid())


    def set_status_handler(self, status_handler):
        """Set or update the status handler for UI updates"""
        self.status_handler = status_handler

    def update_status(self, message: str, restart_counter: bool = False):
        if restart_counter:
            self.progress_counter = 0
        self.progress_counter += 1
        if "progress_bar" in self.status_handler:
            self.status_handler["progress_bar"].progress(int((self.progress_counter / self.progress_max_count) * 100))

        """Update status in the UI if status_handler is provided"""
        if "status_container" in self.status_handler:
            if "blinking_text" in self.status_handler:
                self.status_handler["status_container"].html(self.status_handler["blinking_text"](message))
            else:
                self.status_handler["status_container"].text(message)

    
    
    def report_planner(self, state: AgentState) -> AgentState:
        """Plan the report structure"""

        self.update_status("Planning report structure...", restart_counter=True)

        web_snippets = []
        context = ""
        query = state["messages"][-1].content if state["messages"] else state.get("original_query", "")

        try:
            self.update_status("Fetching web results for report planning...")
            web_results = self.web_search_tool.run(query)
            for result in web_results:  # Limit results
                web_snippets.append(result.get("snippet", ""))

            context = "\n\n".join(web_snippets)

        except Exception as e:
            logger.warning(f"Web search for report planning failed: {e}")

        try:

            analysis_prompt = ChatPromptTemplate.from_template(
                """
                You are a research planner. Based on the user's research query, generate a structured outline for a research report.  
                - The report should contain 2-3 sections, beginning with an Introduction and ending with a Conclusion.  
                - For each section, create 2-3 focused sub-queries that are *explicitly and directly* connected to the research query, 
                to guide information gathering.
                - each sub-query must be complete on its own and related to the research query. 
                    example of sub-queries:
                    topic of research: Tom and Jerry
                    bad example of sub-query: "First theatrical short film release?"
                    good example of sub-query: "When was the first theatrical short film of Tom and Jerry released?"
                - also find a suitable title for the report. A report title should be clear, concise, and informative, directly related to the user's query and purpose to help readers immediately understand what the report addresses.
                - Use both the provided context and your own knowledge to ensure comprehensive coverage of the topic.

                context: {context}
                
                Research Query: {query}

                Respond only with a JSON object containing: report_sections, and no other explanation. JSON object should be
                wrapped inside multiline python string. it should not be wrapped inside triple backticks or any other markdown syntax.
                like:
                {{
                    "report_sections": [
                        {{
                            "title": "Section 1 Title",
                            "sub_queries": [
                                "Sub-query 1",
                                "Sub-query 2"
                            ]
                        }},
                        {{
                            "title": "Section 2 Title",
                            "sub_queries": [
                                "Sub-query 1",
                                "Sub-query 2"
                            ]
                        }}
                    ],
                    "report_title": "Title of the Report"
                }}
                """
            )
            self.update_status("Planning report sections...")
            chain = analysis_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"context": context, "query": query})
            try:
                # print("report planning result: ", result)
                plan = json.loads(result)
                # print("plan: ", plan)
                state["report_plan"] = plan
                # Set flag to require human review
                state["human_review_required"] = True
                state["human_approved"] = False
                state["human_feedback"] = ""
                state["modified_plan"] = {}
            except json.JSONDecodeError:
                print("Report planning result is not valid JSON")
                print("result:\n", result)
                state["report_plan"] = {"report_sections": []}

            logger.info("Report planning completed (placeholder)")
        except Exception as e:
            error_msg = f"Report planning failed: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            state["errors"].append(error_msg)

        return state
    
    def human_review_node(self, state: AgentState) -> AgentState:
        """Node for human review of the research plan"""
        self.update_status("Waiting for human review of research plan...")
        
        # This node will pause execution and wait for human input
        # The actual review will be handled by the UI
        # For now, we just mark that review is needed
        state["human_review_required"] = True

        
        # If there's a modified plan from human feedback, use it
        if state.get("modified_plan") and state["modified_plan"]:
            state["report_plan"] = state["modified_plan"]
            logger.info("Using human-modified research plan")

        return state
    
    def should_continue_to_generation(self, state: AgentState) -> str:
        """Conditional edge function to determine next step after human review"""
        if state.get("human_approved", False):
            return "report_generator"
        else:
            # Stay in review mode until human approves
            return "report_planner"
    
    def report_generator(self, state: AgentState) -> AgentState:
        """Generate the report based on the plan"""
        # Placeholder implementation
        self.update_status("Generating report...")  
        try:
            plan = state.get("report_plan", {})
            report_sections = plan.get("report_sections", [])
            report_title = plan.get("report_title", "Research Report")
            report_content = f"## {report_title}\n\n"
            self.progress_max_count += len(report_sections)
            for i, section in enumerate(report_sections):
                title = section.get("title", "Untitled Section")
                report_content += f"### {i+1}. {title}\n\n"
                combined_snippets = ""
                self.update_status(f"Generating content for section ({i+1}/{len(report_sections)}): {title}")
                for sub_query in section.get("sub_queries", []):
                    try:
                        # Use web search tool to fetch content for the sub-query
                        web_results = self.web_search_tool.run(sub_query)
                        snippets = [result.get("snippet", "") for result in web_results]  # Collect top 3 snippets
                        combined_snippets = "\n".join(snippets)
                    except Exception as e:
                        logger.warning(f"Web search failed for sub-query '{sub_query}': {e}")

                # summarize the snippets using LLM   
                try:
                    # Use LLM to generate content based on the combined snippets
                    content_prompt = ChatPromptTemplate.from_template(
                        """
                    Based on the following snippets, generate a detailed and coherent section-content for the below
                    section-heading and research-topic. the content should be comprehensive and informative. the content should be 
                    20-30 words long, written in natural language, and in paragraph form.

                    Snippets:
                    {snippets}

                    Section-heading: {section_heading}
                    Research-topic: {topic}

                    Respond with the generated content for this section and do not add the section-heading.
                    """)
                    chain = content_prompt | self.llm | StrOutputParser()
                    section_content = chain.invoke({"snippets": combined_snippets, "section_heading": title, "topic": report_title})
                    report_content += f"{section_content}\n"
                except Exception as e:
                    logger.warning(f"LLM content generation failed for this section: '{title}': {e}")
                    report_content += f"  - Error generating content for this section.\n"

                report_content += "\n"
            self.update_status("Report generation completed.")
            state["generated_report"] = report_content
            logger.info("Report generation completed (placeholder)")
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["generated_report"] = "# Research Report\n\nError generating report."
        return state

    
    def _format_docs(self, docs):
        """Format retrieved documents for RAG context"""
        return "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph using standard LangGraph patterns"""
        
        # Create workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes for each step in the workflow
        workflow.add_node("report_planner", self.report_planner)
        workflow.add_node("human_review_node", self.human_review_node)
        workflow.add_node("report_generator", self.report_generator)

        # Define edges between nodes
        workflow.add_edge(START, "report_planner")
        workflow.add_edge("report_planner", "human_review_node")
        
        # Add conditional edge from human review
        workflow.add_conditional_edges(
            "human_review_node",
            self.should_continue_to_generation,
            {
                "report_generator": "report_generator",
                "report_planner": "report_planner"  # Loop back if not approved
            }
        )
        
        workflow.add_edge("report_generator", END)
        
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review_node"]
        )
    
    def _log_step(self, step_name: str, inputs: Any, outputs: Any, execution_time: float, state: Dict[str, Any]):
        """Log step execution details"""
        step_log = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'status': 'success' if not state.get('errors') else 'error',
            'input_summary': str(inputs)[:200],
            'output_summary': str(outputs)[:200],
        }
        
        if 'step_logs' not in state:
            state['step_logs'] = []
        if 'execution_times' not in state:
            state['execution_times'] = {}
            
        state['step_logs'].append(step_log)
        state['execution_times'][step_name] = execution_time
        
        logger.info(f"Step '{step_name}' completed in {execution_time:.2f}s")
    
    def run(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Execute the research workflow with standard LangGraph approach"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Initialize state with standard LangGraph message format
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "query_analysis": {},
                "raw_documents": [],
                "retrieved_docs": [],
                "generated_text": "",
                "markdown_report": "",
                "timestamp": "",
                "errors": [],
                "step_logs": [],
                "execution_times": {},
                "human_review_required": False,
                "human_approved": False,
                "human_feedback": "",
                "modified_plan": {}
            }
            
            # Execute the graph with config
            result = self.graph.invoke(initial_state, config)

            with open("debug_log", "w") as f:
                f.write(str(result))
                # f.write(json.dumps(result, indent=2))
            
            logger.info(f"Research workflow completed for query: {query}")
            return result
            
        except Exception as e:
            # Check if this is an interruption at human review
            if "interrupt" in str(e).lower() or "human_review" in str(e).lower():
                logger.info("Workflow interrupted for human review")
                # Return current state for review
                try:
                    state = self.get_current_state(thread_id)
                    if state:
                        return state
                except:
                    pass
            
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            return {
                "original_query": query,
                "generated_text": f"Research failed: {str(e)}",
                "markdown_report": f"# Research Failed\n\nError: {str(e)}",
                "errors": [error_msg],
                "timestamp": datetime.now().isoformat(),
                "step_logs": [],
                "execution_times": {},
                "human_review_required": False,
                "human_approved": False,
                "human_feedback": "",
                "modified_plan": {}
            }
    
    def get_current_state(self, thread_id: str = "default") -> Dict[str, Any]:
        """Get the current state of the workflow"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            return state.values if state else {}
        except Exception as e:
            logger.error(f"Failed to get current state: {e}")
            return {}
    
    def approve_plan(self, thread_id: str = "default", modified_plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """Approve the research plan and continue execution"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Update state with approval
            updates = {
                "human_approved": True,
                "human_review_required": False
            }
            
            if modified_plan:
                updates["modified_plan"] = modified_plan
                updates["human_feedback"] = "Plan modified by human"
            else:
                updates["human_feedback"] = "Plan approved without changes"
            
            # Update the state and continue
            self.graph.update_state(config, updates)
            
            # Continue execution from where it left off
            final_result = self.graph.invoke(None, config)
            
            return final_result
            
        except Exception as e:
            error_msg = f"Failed to approve plan: {str(e)}"
            logger.error(error_msg)
            return {"errors": [error_msg]}
    
    def reject_plan(self, thread_id: str = "default", feedback: str = "") -> Dict[str, Any]:
        """Reject the plan and request regeneration"""
        print("="*20, "[Graph] Rejecting Plan ", "="*20)
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            updates = {
                "human_approved": False,
                "human_feedback": feedback,
                "human_review_required": False
            }
            
            self.graph.update_state(config, updates)
            final_result = self.graph.invoke(None, config)
            return final_result
            
        except Exception as e:
            error_msg = f"Failed to reject plan: {str(e)}"
            logger.error(error_msg)
            return {"error": [error_msg]}
