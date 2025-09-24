"""
Graph Builder - Flexible workflow construction for research agent

This module provides utilities for building and configuring LangGraph workflows
in a modular, extensible way.
"""
from typing import Dict, Any, List, Type, Optional, Callable
import logging
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state_manager import AgentState
from .nodes.base_node import BaseNode, ConditionalNode

logger = logging.getLogger(__name__)


class NodeConfig:
    """Configuration for a single node in the workflow"""
    
    def __init__(
        self, 
        node_class: Type[BaseNode],
        name: str,
        dependencies: Dict[str, Any] = None,
        config_overrides: Dict[str, Any] = None
    ):
        self.node_class = node_class
        self.name = name
        self.dependencies = dependencies or {}
        self.config_overrides = config_overrides or {}


class EdgeConfig:
    """Configuration for edges between nodes"""
    
    def __init__(
        self, 
        from_node: str, 
        to_node: str, 
        condition_func: Optional[Callable] = None,
        edge_mapping: Optional[Dict[str, str]] = None
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.condition_func = condition_func
        self.edge_mapping = edge_mapping or {}


class WorkflowBuilder:
    """Builder for creating LangGraph workflows with configurable nodes and edges"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.nodes: Dict[str, NodeConfig] = {}
        self.edges: List[EdgeConfig] = []
        self.entry_node: Optional[str] = None
        self.exit_nodes: List[str] = []
        self.interrupt_before: List[str] = []
        self.checkpointer = MemorySaver()
        
    def add_node(
        self, 
        name: str, 
        node_class: Type[BaseNode],
        dependencies: Dict[str, Any] = None,
        config_overrides: Dict[str, Any] = None
    ) -> 'WorkflowBuilder':
        """Add a node to the workflow"""
        self.nodes[name] = NodeConfig(node_class, name, dependencies, config_overrides)
        return self
    
    def add_edge(
        self, 
        from_node: str, 
        to_node: str, 
        condition_func: Optional[Callable] = None,
        edge_mapping: Optional[Dict[str, str]] = None
    ) -> 'WorkflowBuilder':
        """Add an edge between nodes"""
        self.edges.append(EdgeConfig(from_node, to_node, condition_func, edge_mapping))
        return self
    
    def set_entry_node(self, node_name: str) -> 'WorkflowBuilder':
        """Set the entry point for the workflow"""
        self.entry_node = node_name
        return self
    
    def add_exit_node(self, node_name: str) -> 'WorkflowBuilder':
        """Add an exit node (nodes that lead to END)"""
        self.exit_nodes.append(node_name)
        return self
    
    def add_interrupt(self, node_name: str) -> 'WorkflowBuilder':
        """Add a node that should interrupt for human input"""
        self.interrupt_before.append(node_name)
        return self
    
    def build(self) -> StateGraph:
        """Build the LangGraph StateGraph from the configuration"""
        if not self.nodes:
            raise ValueError("No nodes configured for workflow")
        
        if not self.entry_node:
            raise ValueError("No entry node configured")
        
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Initialize and add nodes
        node_instances = {}
        for name, node_config in self.nodes.items():
            # Merge base config with node-specific overrides
            merged_config = {**self.base_config, **node_config.config_overrides}
            
            # Create node instance with dependencies
            node_instance = node_config.node_class(
                config=merged_config,
                **node_config.dependencies
            )
            
            node_instances[name] = node_instance
            workflow.add_node(name, node_instance)
        
        # Add entry edge
        workflow.add_edge(START, self.entry_node)
        
        # Add configured edges
        for edge_config in self.edges:
            if edge_config.condition_func and edge_config.edge_mapping:
                # Conditional edge
                workflow.add_conditional_edges(
                    edge_config.from_node,
                    edge_config.condition_func,
                    edge_config.edge_mapping
                )
            else:
                # Simple edge
                workflow.add_edge(edge_config.from_node, edge_config.to_node)
        
        # Add exit edges
        for exit_node in self.exit_nodes:
            workflow.add_edge(exit_node, END)
        
        # Compile with checkpointer and interrupts
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=self.interrupt_before if self.interrupt_before else None
        )


class ResearchWorkflowFactory:
    """Factory for creating pre-configured research workflows"""
    
    @staticmethod
    def create_standard_workflow(
        config: Dict[str, Any],
        llm_provider,
        web_search_tool,
        status_handler=None
    ) -> StateGraph:
        """Create the standard research agent workflow"""
        from .nodes.report_planner import ReportPlannerNode
        from .nodes.human_review import HumanReviewNode
        from .nodes.report_generator import ReportGeneratorNode
        
        builder = WorkflowBuilder(config)
        
        # Configure nodes with their dependencies
        builder.add_node(
            "report_planner",
            ReportPlannerNode,
            dependencies={
                "llm_provider": llm_provider,
                "web_search_tool": web_search_tool,
                "status_handler": status_handler
            }
        )
        
        builder.add_node(
            "human_review_node",
            HumanReviewNode,
            dependencies={"status_handler": status_handler}
        )
        
        builder.add_node(
            "report_generator",
            ReportGeneratorNode,
            dependencies={
                "llm_provider": llm_provider,
                "web_search_tool": web_search_tool,
                "status_handler": status_handler
            }
        )
        
        # Configure workflow structure
        builder.set_entry_node("report_planner")
        
        builder.add_edge("report_planner", "human_review_node")
        
        # Conditional edge from human review
        def should_continue_to_generation(state: AgentState) -> str:
            """Determine next step after human review"""
            if state.get("human_approved", False):
                return "report_generator"
            else:
                return "report_planner"
        
        builder.add_edge(
            "human_review_node",
            "conditional",  # This gets resolved by the conditional mapping
            condition_func=should_continue_to_generation,
            edge_mapping={
                "report_generator": "report_generator",
                "report_planner": "report_planner"
            }
        )
        
        builder.add_exit_node("report_generator")
        builder.add_interrupt("human_review_node")
        
        return builder.build()
    
    @staticmethod
    def create_custom_workflow(
        config: Dict[str, Any],
        node_configurations: List[Dict[str, Any]],
        edge_configurations: List[Dict[str, Any]],
        **dependencies
    ) -> StateGraph:
        """Create a custom workflow from configuration"""
        builder = WorkflowBuilder(config)
        
        # Add nodes from configuration
        for node_config in node_configurations:
            builder.add_node(
                name=node_config["name"],
                node_class=node_config["class"],
                dependencies=node_config.get("dependencies", {}),
                config_overrides=node_config.get("config", {})
            )
        
        # Add edges from configuration
        for edge_config in edge_configurations:
            builder.add_edge(
                from_node=edge_config["from"],
                to_node=edge_config["to"],
                condition_func=edge_config.get("condition"),
                edge_mapping=edge_config.get("mapping")
            )
        
        # Set workflow structure from config
        builder.set_entry_node(node_configurations[0]["name"])  # First node as entry
        
        return builder.build()


def validate_workflow_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate workflow configuration"""
    errors = []
    
    # Check required fields
    if "nodes" not in config:
        errors.append("Missing 'nodes' configuration")
    
    if "edges" not in config:
        errors.append("Missing 'edges' configuration")
    
    # Validate node references in edges
    if "nodes" in config and "edges" in config:
        node_names = {node["name"] for node in config["nodes"]}
        
        for edge in config["edges"]:
            if edge["from"] not in node_names:
                errors.append(f"Edge references unknown node: {edge['from']}")
            if edge["to"] not in node_names and edge["to"] != "END":
                errors.append(f"Edge references unknown node: {edge['to']}")
    
    return len(errors) == 0, errors