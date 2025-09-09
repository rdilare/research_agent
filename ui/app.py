"""
Streamlit UI for Research Assistant Agent using Standard LangChain/LangGraph
"""
import streamlit as st
import logging
import yaml
import os
from datetime import datetime
from typing import Dict, Any

# Import the standard LangGraph agent
from agents.graph import ResearchAgentGraph

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from yaml file"""
    config_path = "config/settings.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration for standard LangChain setup
        return {
            "llm": {
                "model": "llama3.2",
                "base_url": "http://localhost:11434"
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2"
            },
            "logging": {
                "level": "INFO"
            }
        }


def blinking_text(text, should_blink=True):
    
    html_code = f"""
    <span style="font-size:14px; color: grey; font-weight: normal; animation: blinker 1s linear infinite;">
        {text}
    </span>
    <style>
    @keyframes blinker {{
        20% {{ opacity: 0; }}
    }}
    </style>
    """
        
    return html_code

def display_debug_info(results):
    """Display detailed debug information"""
    st.subheader("🐛 Debug Information")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Enable detailed debug mode", help="Shows full input/output data")
    
    if results.get('step_logs'):
        st.subheader("Step Execution Log")
        
        for step_log in results['step_logs']:
            with st.expander(f"📋 {step_log['step']} - {step_log['status'].upper()}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Execution Time", f"{step_log['execution_time']:.2f}s")
                    st.write(f"**Status:** {step_log['status']}")
                    st.write(f"**Timestamp:** {step_log['timestamp']}")
                
                with col2:
                    st.write("**Input Summary:**")
                    st.code(step_log.get('input_summary', 'No input data'), language='json')
                
                with col3:
                    st.write("**Output Summary:**")
                    st.code(step_log.get('output_summary', 'No output data'), language='json')
                
                if step_log.get('errors'):
                    st.error("Errors encountered:")
                    for error in step_log['errors']:
                        st.write(f"• {error}")
                
                if debug_mode and (step_log.get('full_input') or step_log.get('full_output')):
                    st.write("**Detailed Data (Debug Mode):**")
                    if step_log.get('full_input'):
                        with st.expander("Full Input Data"):
                            st.text(step_log['full_input'])
                    if step_log.get('full_output'):
                        with st.expander("Full Output Data"):
                            st.text(step_log['full_output'])
    
    # Execution time overview
    if results.get('execution_times'):
        st.subheader("⏱️ Performance Overview")
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        steps = list(results['execution_times'].keys())
        times = list(results['execution_times'].values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(steps, times, color='skyblue', alpha=0.7)
        ax.set_xlabel('Pipeline Steps')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Step Execution Times')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Performance metrics
        total_time = sum(times)
        st.metric("Total Execution Time", f"{total_time:.2f}s")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fastest Step", f"{min(times):.2f}s")
        with col2:
            st.metric("Slowest Step", f"{max(times):.2f}s")
        with col3:
            st.metric("Average Step Time", f"{np.mean(times):.2f}s")
    
    # Step input/output details
    if results.get('step_inputs') and results.get('step_outputs'):
        st.subheader("📊 Step Data Flow")
        
        step_names = list(results['step_inputs'].keys())
        selected_step = st.selectbox("Select step to inspect:", step_names)
        
        if selected_step:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Input to {selected_step}:**")
                if selected_step in results['step_inputs']:
                    st.code(results['step_inputs'][selected_step], language='json')
                else:
                    st.write("No input data available")
            
            with col2:
                st.write(f"**Output from {selected_step}:**")
                if selected_step in results['step_outputs']:
                    st.code(results['step_outputs'][selected_step], language='json')
                else:
                    st.write("No output data available")
    
    # Raw results inspection
    st.subheader("🔍 Raw Results Inspection")
    if debug_mode:
        st.write("**Complete Results Object:**")
        st.code(str(results), language='json')
    else:
        st.info("Enable detailed debug mode to see raw results")


def configure_logging():
    """Configure logging based on settings"""
    config = load_config()
    log_level = config.get('logging', {}).get('level', 'INFO')
    debug_mode = config.get('logging', {}).get('debug_mode', False)
    
    if debug_mode:
        log_level = 'DEBUG'
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return debug_mode


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Research Assistant Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configure logging
    debug_mode = configure_logging()
    
    # Load configuration
    config = load_config()
    
    # Initialize session state
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    # Sidebar
    with st.sidebar:
        st.title("Research Assistant")
        st.markdown("---")
        
        # Configuration options
        st.subheader("Configuration")
        
        # Data source selection
        st.write("**Data Sources**")
        use_arxiv = st.checkbox("ArXiv Papers", value=config['data_sources']['arxiv']['enabled'])
        use_duckduckgo = st.checkbox("Web Search", value=config['data_sources']['duckduckgo']['enabled'])
        
        # Content extraction options
        if use_duckduckgo:
            extract_content = st.checkbox(
                "Extract Full Content", 
                value=config['data_sources']['duckduckgo'].get('extract_full_content', True),
                help="Extract full webpage content instead of just snippets (slower but more comprehensive)"
            )
            if extract_content:
                config['data_sources']['duckduckgo']['extract_full_content'] = True
            else:
                config['data_sources']['duckduckgo']['extract_full_content'] = False
        
        # Research parameters
        st.write("**Parameters**")
        max_sources = st.slider("Max Sources", 5, 50, 10)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7)
        
        st.markdown("---")
        
        # Debug and logging configuration
        st.subheader("Debug & Logging")
        debug_enabled = st.checkbox("Enable Debug Mode", value=debug_mode, help="Enables detailed logging and debug information")
        step_logging = st.checkbox("Step-by-step Logging", value=True, help="Log each processing step")
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        
        # Update config with debug settings
        if debug_enabled:
            config['logging']['debug_mode'] = True
            config['logging']['step_logging'] = step_logging
            config['logging']['level'] = log_level
        else:
            config['logging']['debug_mode'] = False
        
        st.markdown("---")
        
        # Research history
        st.subheader("Research History")
        if st.session_state.research_history:
            for i, query in enumerate(st.session_state.research_history[-5:], 1):
                if st.button(f"{i}. {query[:30]}...", key=f"history_{i}"):
                    st.session_state.current_query = query
        else:
            st.write("No previous research")
        
        if st.button("Clear History"):
            st.session_state.research_history = []
            st.rerun()
    
    # Main content area
    st.title("Research Assistant Agent")
    st.markdown("*Comprehensive research across all domains with AI-powered analysis*")
    
    # Query input
    st.subheader("Research Query")
    user_query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the latest trends in artificial intelligence for healthcare?",
        height=100,
        value=st.session_state.get('current_query', '')
    )
    
    # Research controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Start Research", type="primary", disabled=not user_query.strip()):
            if user_query.strip():
                conduct_research(user_query, config)
    
    with col2:
        if st.button("💾 Save Results") and st.session_state.current_results:
            save_results(st.session_state.current_results)
    
    # Display results
    if st.session_state.current_results:
        display_results(st.session_state.current_results)


def conduct_research(query: str, config: Dict[str, Any]):
    """Conduct research using the agent graph"""
    try:
        # Update configuration with UI selections
        # This would update the config based on user selections
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.html(blinking_text("Initializing research agent..."))
        progress_bar.progress(10)
        
        # Initialize research agent
        agent = ResearchAgentGraph(config)
        
        status_text.html(blinking_text("Analyzing query..."))
        progress_bar.progress(20)
        
        # Run research workflow
        status_text.html(blinking_text("Retrieving data from sources..."))
        progress_bar.progress(40)
        
        results = agent.run(query)
        
        status_text.html(blinking_text("Processing and analyzing data..."))
        progress_bar.progress(70)
        
        status_text.html(blinking_text("Generating insights..."))
        progress_bar.progress(90)
        
        # Generate the markdown report
        status_text.html(blinking_text("Preparing report..."))
        progress_bar.progress(95)
        
        # Store results
        st.session_state.current_results = results
        st.session_state.research_history.append(query)
        
        # Add metadata to results for report generation
        if results:
            import datetime
            results['original_query'] = query
            results['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate the markdown report automatically
        if results:
            markdown_report = generate_report(results)
            # Store the markdown report in the results for easy access
            results['markdown_report'] = markdown_report
        
        status_text.text("Research completed!")
        progress_bar.progress(100)
        
        st.success("Research completed successfully!")
        
    except Exception as e:
        st.error(f"Research failed: {str(e)}")
        logger.error(f"Research failed: {e}")


def display_results(results):
    """Display research results in a simple markdown format"""
    try:
        st.subheader("📋 Research Report")
        
        # Display the auto-generated markdown report first
        if results.get('markdown_report'):
            # Display the full markdown report
            st.markdown(results['markdown_report'])
            
            # Add download button for the report
            st.download_button(
                label="📄 Download Report as Markdown",
                data=results['markdown_report'],
                file_name=f"research_report_{results.get('timestamp', 'unknown')}.md",
                mime="text/markdown",
                help="Download the complete research report as a markdown file"
            )
            
        elif results.get('generated_text'):
            # Fallback to displaying just the analysis if no full report
            st.markdown("---")
            st.markdown("## Research Analysis")
            st.markdown(results['generated_text'])
        else:
            st.warning("Research analysis not available")
        
        # Add debug info in collapsed section for troubleshooting
        if results.get('step_logs'):
            with st.expander("🐛 Debug Information", expanded=False):
                display_debug_info(results)
        
    except Exception as e:
        st.error(f"Failed to display results: {str(e)}")
        logger.error(f"Display results error: {e}")


def generate_report(results):
    """Generate markdown formatted report"""
    try:
        # Build markdown report
        report_parts = []
        
        # Title and metadata
        report_parts.append("# Research Report")
        report_parts.append(f"*Generated on: {results.get('timestamp', 'Unknown')}*\n")
        
        # Query information
        if results.get('original_query'):
            report_parts.append(f"**Research Query:** {results['original_query']}\n")
        
        # Main analysis
        if results.get('generated_text'):
            report_parts.append("## Analysis")
            report_parts.append(results['generated_text'])
            report_parts.append("")
        
        # Sources summary
        if results.get('raw_data'):
            report_parts.append("## Sources")
            report_parts.append(f"This analysis is based on {len(results['raw_data'])} sources:")
            
            # Group by source type
            source_types = {}
            for item in results['raw_data']:
                source = item.get('source', 'unknown')
                source_types[source] = source_types.get(source, 0) + 1
            
            for source, count in source_types.items():
                report_parts.append(f"- {source.title()}: {count} documents")
            
            report_parts.append("")
            
            # Detailed source list
            report_parts.append("### Detailed Sources")
            for i, item in enumerate(results['raw_data'][:10], 1):  # Show top 10 sources
                title = item.get('title', 'Unknown Title')
                url = item.get('url', '')
                source = item.get('source', 'Unknown')
                
                if url:
                    report_parts.append(f"{i}. [{title}]({url}) - *{source}*")
                else:
                    report_parts.append(f"{i}. {title} - *{source}*")
        
        # Join all parts
        markdown_report = "\n".join(report_parts)
        
        return markdown_report
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return f"# Error\nFailed to generate report: {str(e)}"


def save_results(results):
    """Save research results"""
    try:
        st.info("Results saved to session!")
        # This would implement proper results saving
        
    except Exception as e:
        st.error(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    main()
