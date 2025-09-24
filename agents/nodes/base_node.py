"""
Base class for all graph nodes in the research agent workflow
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging
from ..state_manager import AgentState, StateManager

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Abstract base class for all workflow nodes"""
    
    def __init__(self, config: Dict[str, Any], status_handler: Optional[Dict[str, Any]] = None):
        self.config = config
        self.status_handler = status_handler
        self.node_name = self.__class__.__name__
        
    def set_status_handler(self, status_handler: Dict[str, Any]):
        """Set or update the status handler for UI updates"""
        self.status_handler = status_handler
        
    def update_status(self, message: str):
        """Update status in the UI if status_handler is provided"""
        if not self.status_handler or not isinstance(self.status_handler, dict):
            return
            
        if "status_container" in self.status_handler:
            if "blinking_text" in self.status_handler:
                self.status_handler["status_container"].html(
                    self.status_handler["blinking_text"](message)
                )
            else:
                self.status_handler["status_container"].text(message)
    
    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Execute the node logic and return updated state"""
        pass
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Wrapper method that handles common node execution patterns:
        - Timing
        - Error handling  
        - Logging
        - Status updates
        """
        start_time = time.time()
        
        try:
            # Update status at start
            self.update_status(f"Executing {self.node_name}...")
            
            # Execute the main node logic
            updated_state = self.execute(state)
            
            # Log successful execution
            exec_time = time.time() - start_time
            StateManager.log_step(
                updated_state,
                self.node_name,
                self._get_input_summary(state),
                self._get_output_summary(updated_state),
                exec_time
            )
            
            logger.info(f"Node '{self.node_name}' completed successfully in {exec_time:.2f}s")
            return updated_state
            
        except Exception as e:
            # Handle errors gracefully
            exec_time = time.time() - start_time
            error_msg = f"Node '{self.node_name}' failed: {str(e)}"
            
            logger.error(error_msg)
            StateManager.add_error(state, error_msg)
            
            # Log failed execution
            StateManager.log_step(
                state,
                self.node_name,
                self._get_input_summary(state),
                {"error": error_msg},
                exec_time
            )
            
            # Update status with error
            self.update_status(f"Error in {self.node_name}: {str(e)}")
            
            return state
    
    def _get_input_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate a summary of inputs for logging"""
        return {
            "query": StateManager.get_query(state)[:100],
            "has_plan": bool(state.get("report_plan")),
            "error_count": len(state.get("errors", []))
        }
    
    def _get_output_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate a summary of outputs for logging"""
        return {
            "has_plan": bool(state.get("report_plan")),
            "has_report": bool(state.get("generated_report")),
            "error_count": len(state.get("errors", []))
        }


class ConditionalNode(BaseNode):
    """Base class for nodes that implement conditional logic"""
    
    @abstractmethod
    def get_next_node(self, state: AgentState) -> str:
        """Determine the next node to execute based on state"""
        pass
    
    def execute(self, state: AgentState) -> AgentState:
        """Default implementation that just returns state unchanged"""
        return state