"""
State Management Utilities for Research Agent

This module provides utilities for managing and manipulating the AgentState
throughout the workflow execution.
"""
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from datetime import datetime
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Standard LangGraph state definition using TypedDict"""
    messages: List[Any]  # Using add_messages annotation from graph.py
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


class StateManager:
    """Utility class for managing agent state transitions and updates"""
    
    @staticmethod
    def create_initial_state(query: str) -> AgentState:
        """Create initial state for a new workflow execution"""
        return {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "query_analysis": {},
            "raw_documents": [],
            "retrieved_docs": [],
            "generated_text": "",
            "markdown_report": "",
            "timestamp": datetime.now().isoformat(),
            "report_plan": {},
            "generated_report": "",
            "errors": [],
            "step_logs": [],
            "execution_times": {},
            "human_review_required": False,
            "human_approved": False,
            "human_feedback": "",
            "modified_plan": {}
        }
    
    @staticmethod
    def add_error(state: AgentState, error: str) -> AgentState:
        """Add an error to the state"""
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(error)
        logger.error(f"State error added: {error}")
        return state
    
    @staticmethod
    def log_step(
        state: AgentState, 
        step_name: str, 
        inputs: Any, 
        outputs: Any, 
        execution_time: float
    ) -> AgentState:
        """Log step execution details in the state"""
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
        return state
    
    @staticmethod
    def set_human_review(
        state: AgentState, 
        required: bool = True, 
        approved: Optional[bool] = None, 
        feedback: str = ""
    ) -> AgentState:
        """Update human review fields in the state"""
        state["human_review_required"] = required
        if approved is not None:
            state["human_approved"] = approved
        if feedback:
            state["human_feedback"] = feedback
        return state
    
    @staticmethod
    def update_plan(state: AgentState, plan: Dict[str, Any], is_modified: bool = False) -> AgentState:
        """Update the report plan in the state"""
        if is_modified:
            state["modified_plan"] = plan
            state["human_feedback"] = "Plan modified by human"
        else:
            state["report_plan"] = plan
        return state
    
    @staticmethod
    def get_query(state: AgentState) -> str:
        """Extract the current query from state messages or original_query"""
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
        return state.get("original_query", "")
    
    @staticmethod
    def get_plan_sections(state: AgentState) -> List[Dict[str, Any]]:
        """Get report sections from current or modified plan"""
        plan = state.get("modified_plan") or state.get("report_plan", {})
        return plan.get("report_sections", [])
    
    @staticmethod
    def get_report_title(state: AgentState) -> str:
        """Get report title from current or modified plan"""
        plan = state.get("modified_plan") or state.get("report_plan", {})
        return plan.get("report_title", "Research Report")
    
    @staticmethod
    def is_state_valid(state: AgentState) -> tuple[bool, List[str]]:
        """Validate state completeness and return validation results"""
        validation_errors = []
        
        # Check required fields
        required_fields = ["messages", "original_query", "errors", "step_logs", "execution_times"]
        for field in required_fields:
            if field not in state:
                validation_errors.append(f"Missing required field: {field}")
        
        # Check data types
        if "messages" in state and not isinstance(state["messages"], list):
            validation_errors.append("messages must be a list")
        
        if "errors" in state and not isinstance(state["errors"], list):
            validation_errors.append("errors must be a list")
        
        return len(validation_errors) == 0, validation_errors
    
    @staticmethod
    def safe_state_copy(state: AgentState) -> AgentState:
        """Create a safe deep copy of the state"""
        try:
            return deepcopy(state)
        except Exception as e:
            logger.warning(f"Deep copy failed, using shallow copy: {e}")
            return state.copy()


class StateValidationError(Exception):
    """Raised when state validation fails"""
    pass


def validate_state_transition(from_node: str, to_node: str, state: AgentState) -> bool:
    """
    Validate that a state transition is allowed based on current state
    
    Args:
        from_node: Source node name
        to_node: Target node name  
        state: Current agent state
        
    Returns:
        True if transition is valid, False otherwise
    """
    # Define valid transitions
    valid_transitions = {
        "report_planner": ["human_review_node"],
        "human_review_node": ["report_generator", "report_planner"],
        "report_generator": ["END"]
    }
    
    if from_node not in valid_transitions:
        logger.warning(f"Unknown source node: {from_node}")
        return False
    
    if to_node not in valid_transitions[from_node] and to_node != "END":
        logger.warning(f"Invalid transition from {from_node} to {to_node}")
        return False
    
    # Additional state-based validation
    if to_node == "report_generator":
        if not state.get("human_approved", False):
            logger.warning("Cannot proceed to report_generator without human approval")
            return False
    
    return True