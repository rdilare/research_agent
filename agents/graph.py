"""
Refactored Research Agent Graph - Modular and Scalable Implementation

This is the refactored version of the research agent that uses:
- Pluggable LLM providers
- Modular node architecture 
- Flexible graph building
- Improved state management
"""
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# LangChain imports (with error handling for missing dependencies)
try:
        LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
try:
    from langchain_core.messages import HumanMessage
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False

# Local imports
from .llm_providers import LLMProviderFactory, create_llm_provider
from .state_manager import AgentState, StateManager
from .graph_builder import ResearchWorkflowFactory
from .nodes import ReportPlannerNode, HumanReviewNode, ReportGeneratorNode
from .tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ResearchAgentGraph:
    """
    Refactored Research Agent with modular architecture
    
    Features:
    - Pluggable LLM providers (Ollama, OpenAI, Anthropic, etc.)
    - Modular node system for easy extension
    - Flexible workflow configuration
    - Improved error handling and state management
    """
    
    def __init__(self, config: Dict[str, Any], status_handler=None):
        self.config = config
        self.status_handler = status_handler
        self.progress_max_count = 8
        self.progress_counter = 0
        
        # Validate dependencies
        self._validate_dependencies()
        
        # Initialize LLM provider
        self.llm_provider = self._initialize_llm_provider()
        
        # Initialize tools
        self.web_search_tool = self._initialize_web_search()
        
        # Build the workflow graph
        self.graph = self._build_workflow()
        
        logger.info(f"ResearchAgentGraph initialized with {self.llm_provider.provider_name} provider")
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain Community not available. Please install: pip install langchain-community"
            )
        
        if not LANGCHAIN_CORE_AVAILABLE:
            raise ImportError(
                "LangChain Core not available. Please install: pip install langchain-core"
            )
    
    def _initialize_llm_provider(self):
        """Initialize the LLM provider based on configuration"""
        llm_config = self.config.get('llm', {})
        
        # Set default provider if not specified
        if 'provider' not in llm_config:
            llm_config['provider'] = 'ollama'
        
        # Set default model if not specified
        if 'model' not in llm_config:
            llm_config['model'] = 'llama3.2'
        
        # Set default base_url for Ollama if not specified
        if llm_config['provider'] == 'ollama' and 'base_url' not in llm_config:
            llm_config['base_url'] = 'http://localhost:11434'
        
        # Set default temperature if not specified
        if 'temperature' not in llm_config:
            llm_config['temperature'] = 0.7
        
        try:
            return create_llm_provider(llm_config)
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise
    
    def _initialize_web_search(self):
        """Initialize web search tool"""
        try:
            return WebSearchTool(max_results=5)
        except Exception as e:
            logger.error(f"Failed to initialize web search tool: {e}")
            raise
    
    def _build_workflow(self):
        """Build the workflow using the graph builder"""
        try:
            return ResearchWorkflowFactory.create_standard_workflow(
                config=self.config,
                llm_provider=self.llm_provider,
                web_search_tool=self.web_search_tool,
                status_handler=self.status_handler
            )
        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            raise
    
    def set_status_handler(self, status_handler):
        """Set or update the status handler for UI updates"""
        self.status_handler = status_handler
        
        # Update status handler in all nodes if they support it
        try:
            current_state = self.get_current_state()
            # This is a best-effort update - nodes will use the new handler on next execution
        except:
            pass  # Ignore errors during status handler update
    
    def update_status(self, message: str, restart_counter: bool = False):
        """Update status in the UI (backward compatibility)"""
        if not self.status_handler or not isinstance(self.status_handler, dict):
            return
            
        if restart_counter:
            self.progress_counter = 0
            
        self.progress_counter += 1
        
        if "progress_bar" in self.status_handler:
            progress_percent = min(100, int((self.progress_counter / self.progress_max_count) * 100))
            self.status_handler["progress_bar"].progress(progress_percent)
        
        if "status_container" in self.status_handler:
            if "blinking_text" in self.status_handler:
                self.status_handler["status_container"].html(
                    self.status_handler["blinking_text"](message)
                )
            else:
                self.status_handler["status_container"].text(message)
    
    def run(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Execute the research workflow"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            
            # Create initial state
            initial_state = StateManager.create_initial_state(query)
            
            # Execute the workflow
            result = self.graph.invoke(initial_state, config)
            
            # Debug logging
            with open("debug_log.log", "w") as f:
                f.write(str(result))
            
            logger.info(f"Research workflow completed for query: {query}")
            return result
            
        except Exception as e:
            # Check if this is an interruption at human review
            if "interrupt" in str(e).lower() or "human_review" in str(e).lower():
                logger.info("Workflow interrupted for human review")
                try:
                    state = self.get_current_state(thread_id)
                    if state:
                        return state
                except:
                    pass
            
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(query, error_msg)
    
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
        """Reject the research plan and request modifications"""
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
            return {"errors": [error_msg]}
    
    def _create_error_result(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            "original_query": query,
            "generated_text": f"Research failed: {error_msg}",
            "markdown_report": f"# Research Failed\n\nError: {error_msg}",
            "errors": [error_msg],
            "timestamp": datetime.now().isoformat(),
            "step_logs": [],
            "execution_times": {},
            "human_review_required": False,
            "human_approved": False,
            "human_feedback": "",
            "modified_plan": {},
            "report_plan": {},
            "generated_report": ""
        }
    
    # Additional methods for extensibility
    
    def add_custom_node(self, node_name: str, node_class, dependencies: Dict[str, Any] = None):
        """Add a custom node to the workflow (for future extension)"""
        # This would require rebuilding the graph - placeholder for future implementation
        raise NotImplementedError("Custom node addition requires workflow rebuild - feature coming soon")
    
    def get_available_providers(self) -> list:
        """Get list of available LLM providers"""
        return LLMProviderFactory.get_available_providers()
    
    def switch_provider(self, provider_config: Dict[str, Any]):
        """Switch to a different LLM provider"""
        try:
            self.llm_provider = create_llm_provider(provider_config)
            # Note: This would require rebuilding the graph for full effect
            logger.info(f"Switched to {self.llm_provider.provider_name} provider")
        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            raise
    
    def get_workflow_structure(self) -> Dict[str, Any]:
        """Get information about the current workflow structure"""
        try:
            # This would analyze the compiled graph structure
            return {
                "nodes": ["report_planner", "human_review_node", "report_generator"],
                "edges": [
                    {"from": "START", "to": "report_planner"},
                    {"from": "report_planner", "to": "human_review_node"},
                    {"from": "human_review_node", "to": "report_generator", "conditional": True},
                    {"from": "report_generator", "to": "END"}
                ],
                "interrupts": ["human_review_node"],
                "provider": self.llm_provider.provider_name
            }
        except Exception as e:
            logger.warning(f"Failed to analyze workflow structure: {e}")
            return {"error": str(e)}


# Backward compatibility - keep the original class name and interface
class ResearchAgentGraphLegacy(ResearchAgentGraph):
    """Legacy wrapper for backward compatibility"""
    pass