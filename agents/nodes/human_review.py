"""
Human Review Node - manages human-in-the-loop workflow for plan approval
"""
import logging
from typing import Dict, Any

from .base_node import ConditionalNode
from ..state_manager import AgentState, StateManager

logger = logging.getLogger(__name__)


class HumanReviewNode(ConditionalNode):
    """Node for human review of the research plan"""
    
    def execute(self, state: AgentState) -> AgentState:
        """Node for human review of the research plan"""
        self.update_status("Waiting for human review of research plan...")
        
        # Set human review as required
        StateManager.set_human_review(state, required=True)
        
        # If human has provided a modified plan, use it
        if state.get("modified_plan") and state["modified_plan"]:
            StateManager.update_plan(state, state["modified_plan"], is_modified=True)
            logger.info("Using human-modified research plan")
        
        return state
    
    def get_next_node(self, state: AgentState) -> str:
        """Conditional edge function to determine next step after human review"""
        if state.get("human_approved", False):
            return "report_generator"
        else:
            # Stay in review mode until human approves
            return "report_planner"
    
    def _get_output_summary(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced output summary for human review"""
        base_summary = super()._get_output_summary(state)
        base_summary.update({
            "approved": state.get("human_approved", False),
            "has_feedback": bool(state.get("human_feedback")),
            "has_modified_plan": bool(state.get("modified_plan")),
            "next_node": self.get_next_node(state)
        })
        return base_summary