"""
Streamlit UI for Research Assistant Agent - Standard LangChain/LangGraph Implementation
"""
import streamlit as st
import logging
import yaml
import os
from datetime import datetime
from typing import Dict, Any

from agents.graph import ResearchAgentGraph

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from yaml file"""
    config_path = "config/settings.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            "llm": {"model": "llama3.2", "base_url": "http://localhost:11434"},
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "data_sources": {
                "arxiv": {"enabled": True},
                "duckduckgo": {"enabled": True}
            },
            "logging": {"level": "INFO"}
        }


def display_results(results: Dict[str, Any]):
    """Display research results"""
    try:
        st.subheader("📋 Research Report")
        
        if results.get('markdown_report'):
            st.markdown(results['markdown_report'])
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=results['markdown_report'],
                file_name=f"research_report_{results.get('timestamp', 'unknown')}.md",
                mime="text/markdown"
            )
        
        elif results.get('generated_text'):
            st.markdown(results['generated_text'])
        else:
            st.warning("No analysis available")
        
        # Debug info
        if results.get('step_logs'):
            with st.expander("🐛 Debug Information"):
                st.write("**Execution Steps:**")
                for step in results['step_logs']:
                    st.write(f"- {step['step']}: {step.get('execution_time', 0):.2f}s")
        
    except Exception as e:
        st.error(f"Display error: {str(e)}")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Research Assistant Agent",
        page_icon="🔬",
        layout="wide"
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = load_config()
    
    # Initialize session state
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    # Main interface
    st.title("🔬 Research Assistant Agent")
    st.markdown("*Standard LangChain & LangGraph Implementation*")
    
    # Query input
    st.subheader("Research Query")
    user_query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the latest developments in AI?",
        height=100
    )
    
    # Research button
    if st.button("🚀 Start Research", type="primary", disabled=not user_query.strip()):
        if user_query.strip():
            with st.spinner("Running research workflow..."):
                try:
                    # Initialize agent
                    agent = ResearchAgentGraph(config)
                    
                    # Run research
                    results = agent.run(user_query)
                    
                    # Store results
                    st.session_state.current_results = results
                    
                    st.success("Research completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Research failed: {str(e)}")
                    logger.error(f"Research error: {e}")
    
    # Display results
    if st.session_state.current_results:
        display_results(st.session_state.current_results)


if __name__ == "__main__":
    main()
