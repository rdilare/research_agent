"""
Streamlit UI for Research Assistant Agent using Standard LangChain/LangGraph
"""
import streamlit as st
import logging
import yaml
import os
from datetime import datetime
from typing import Dict, Any, Optional
import io
import tempfile
from markdown_pdf import MarkdownPdf, Section


# Import the refactored research agent directly
from agents.graph import ResearchAgentGraph

logger = logging.getLogger(__name__)


def markdown_to_pdf(markdown_content: str, title: str = "Document") -> Optional[bytes]:
    """
    Convert markdown text to PDF bytes using markdown-pdf.
    """
    try:
        pdf = MarkdownPdf()
        # Add the whole markdown content as a single section
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.meta["title"] = title
        out = io.BytesIO()
        pdf.save_bytes(out)
        return out.getvalue()  # Returns PDF as bytes
    except Exception as e:
        print(f"Failed to generate PDF: {e}")
        return None


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
                "provider": "ollama",  # Default to Ollama
                "model": "llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0.7
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2"
            },
            "logging": {
                "level": "INFO"
            },
            "search": {
                "max_results": 5
            }
        }


def blinking_text(text, should_blink=True):
    """Create blinking text animation for status updates"""
    html_code = f"""
    <span style="font-size:14px; color: grey; font-weight: normal; animation: blinker 1.5s step-start infinite;">
        {text}
    </span>
    <style>
    @keyframes blinker {{
        20% {{ opacity: 0; }}
    }}
    </style>
    """
        
    return html_code

def display_human_review_interface():
    """Display interface for human review of research plan"""
    st.subheader("🔍 Research Plan Review")
    st.info("Please review the generated research plan below. You can approve it as-is or modify the JSON and then approve.")
    
    try:
        # Get current state from the agent
        current_state = st.session_state.research_agent.get_current_state(st.session_state.thread_id)
        report_plan = current_state.get("report_plan", {})
        
        if not report_plan or not report_plan.get("report_sections"):
            st.warning("No research plan available for review.")
            return

        # JSON editor for the plan
        st.write("**Edit Research Plan (JSON format):**")
        st.write("Modify the JSON below to update the research plan, then click 'Approve Modified Plan'")
        
        import json
        
        # Convert plan to pretty JSON string
        plan_json = json.dumps(report_plan, indent=2)
        
        # Text area for JSON editing
        edited_plan_json = st.text_area(
            "Research Plan JSON:",
            value=plan_json,
            height=400,
            key="plan_json_editor",
            help="Edit the JSON to modify sections, titles, and sub-queries. Make sure to maintain valid JSON format."
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Approve Original Plan", type="primary", key="approve_original"):
                handle_plan_approval(original_plan=True)
        
        with col2:
            if st.button("✅ Approve Modified Plan", type="secondary", key="approve_modified"):
                try:
                    # Parse the edited JSON
                    modified_plan = json.loads(edited_plan_json)
                    handle_plan_approval(modified_plan=modified_plan)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                    st.error("Please fix the JSON syntax and try again.")
        
        with col3:
            if st.button("❌ Reject Plan", type="secondary", key="reject_plan"):
                st.session_state.show_rejection_dialog = True
                st.rerun()
        
        # Show rejection dialog if flag is set
        if st.session_state.get('show_rejection_dialog', False):
            with st.expander("Plan Rejection", expanded=True):
                st.warning("You are about to reject this research plan.")
                
                feedback = st.text_area(
                    "Provide feedback for rejection (optional):",
                    placeholder="e.g., Please add more sections about specific topics, change focus, etc.",
                    key="rejection_feedback_input",
                    height=100
                )
                
                col_confirm, col_cancel = st.columns(2)
                
                with col_confirm:
                    if st.button("🔴 Confirm Rejection", type="primary", key="confirm_reject"):
                        st.session_state.show_rejection_dialog = False
                        handle_plan_rejection(feedback)
                
                with col_cancel:
                    if st.button("↩️ Cancel", type="secondary", key="cancel_reject"):
                        st.session_state.show_rejection_dialog = False
                        st.rerun()

        # JSON validation feedback
        try:
            json.loads(edited_plan_json)
            st.success("✅ JSON format is valid")
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON format error: {str(e)}")
                
    except Exception as e:
        st.error(f"Error displaying review interface: {str(e)}")


def handle_plan_approval(original_plan=False, modified_plan=None):
    """Handle plan approval and continue research"""
    try:
        if original_plan:
            # Approve the original plan
            result = st.session_state.research_agent.approve_plan(st.session_state.thread_id)
        else:
            # Approve with modifications
            result = st.session_state.research_agent.approve_plan(
                st.session_state.thread_id, 
                modified_plan
            )
        
        st.session_state.awaiting_human_review = False
        st.session_state.current_results = result
        
        st.success("Plan approved! Continuing with research...")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error approving plan: {str(e)}")


def handle_plan_rejection(feedback: str = ""):
    """Handle plan rejection and provide feedback options"""
    try:
        # Call the reject_plan method from the graph
        result = st.session_state.research_agent.reject_plan(
            st.session_state.thread_id, 
            feedback if feedback.strip() else "Plan rejected by user"
        )
        
        # Reset the review state
        st.session_state.awaiting_human_review = False
        st.session_state.current_results = result
        
        st.warning("Plan rejected. You can modify your query and restart the research process.")
        if feedback.strip():
            st.info(f"Your feedback: {feedback}")
        st.info("Consider refining your research question or providing more specific requirements.")
                
        st.rerun()
        
    except Exception as e:
        print("="*20, "[UI] Error Rejecting Plan ", "="*20)
        st.error(f"Error rejecting plan: {str(e)}")



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
    
    if 'research_agent' not in st.session_state:
        st.session_state.research_agent = None
        
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = "default"
        
    if 'awaiting_human_review' not in st.session_state:
        st.session_state.awaiting_human_review = False
    
    if 'show_rejection_dialog' not in st.session_state:
        st.session_state.show_rejection_dialog = False
    
    # Sidebar
    with st.sidebar:
        st.title("Research Assistant")
        st.markdown("---")
        
        # Configuration options
        st.subheader("Configuration")
        
        # LLM Provider Selection
        st.write("**LLM Provider**")
        provider_options = ["ollama", "openai", "anthropic"]
        current_provider = config.get("llm", {}).get("provider", "ollama")
        
        selected_provider = st.selectbox(
            "Provider:",
            provider_options,
            index=provider_options.index(current_provider) if current_provider in provider_options else 0,
            help="Select the LLM provider to use"
        )
        
        # Update config based on provider selection
        if selected_provider != current_provider:
            config["llm"]["provider"] = selected_provider
            
            # Reset to provider-specific defaults if switching
            if selected_provider == "ollama":
                config["llm"]["base_url"] = "http://localhost:11434"
                config["llm"]["model"] = "llama3.2"
            elif selected_provider == "openai":
                config["llm"]["model"] = "gpt-4"
                config["llm"].pop("base_url", None)  # Remove base_url for OpenAI
            elif selected_provider == "anthropic":
                config["llm"]["model"] = "claude-3-sonnet"
                config["llm"].pop("base_url", None)  # Remove base_url for Anthropic
        
        
        # Research parameters
        st.write("**Parameters**")
        max_sources = st.slider("Max Sources", 5, 50, 10)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7)
        
        # Update config with UI parameters
        config["llm"]["temperature"] = temperature
        config["max_sources"] = max_sources
        config['logging']['level'] = "ERROR"
        
    
    # Main content area
    st.title("Research Assistant Agent")
    st.markdown("*AI-powered research leveraging the latest information through web search*")

    
    # Query input
    st.subheader("Research Query")
    user_query = st.text_area(
        "Enter your research question:",
        placeholder="e.g., What are the latest trends in artificial intelligence for healthcare?",
        height=100,
        value=st.session_state.get('current_query', '')
    )
    
    # Research controls
    if st.button("🚀 Start Research", type="primary", disabled=False):
        if user_query.strip():
            conduct_research(user_query, config)
    
    
    # Human review interface
    if st.session_state.awaiting_human_review and st.session_state.research_agent:
        display_human_review_interface()
    
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

        status_handler = {
            "status_container": status_text,
            "progress_bar": progress_bar,
            "blinking_text": blinking_text
        }
        
        status_text.html(blinking_text("Initializing research agent..."))
        progress_bar.progress(10)
        
        # Initialize research agent with enhanced error handling
        try:
            agent = ResearchAgentGraph(config, status_handler=status_handler)
            st.session_state.research_agent = agent
            
            # Show provider information
            st.info(f"✓ Initialized with {agent.llm_provider.provider_name.title()} provider")
            
        except ImportError as e:
            st.error("❌ Missing dependencies for refactored implementation")
            st.error(f"Error: {str(e)}")
            st.info("💡 Try installing missing packages or check your environment")
            return
        except Exception as e:
            st.error("❌ Failed to initialize research agent")
            st.error(f"Error: {str(e)}")
            st.info("💡 Check your configuration and ensure the LLM service is running")
            return

        
        # Execute until interrupt (human review)
        try:
            print("="*20, " Starting Research Run ", "="*20)
            results = agent.run(query, st.session_state.thread_id)
            
            # Check if we hit the human review interrupt
            current_state = agent.get_current_state(st.session_state.thread_id)
            if current_state.get("human_review_required", False) and not current_state.get("human_approved", False):
                st.session_state.awaiting_human_review = True
                status_text.text("Research plan generated. Please review below.")
                st.rerun()
                return
            
        except Exception as e:
            # Enhanced error handling for the refactored implementation
            if "interrupt" in str(e).lower() or "human_review" in str(e).lower():
                # This is expected - workflow interrupted for human review
                try:
                    current_state = agent.get_current_state(st.session_state.thread_id)
                    if current_state.get("human_review_required", False):
                        st.session_state.awaiting_human_review = True
                        status_text.text("Research plan generated. Please review below.")
                        st.rerun()
                        return
                except Exception as state_error:
                    logger.error(f"Error getting current state: {state_error}")
            
            # Handle other errors
            logger.error(f"Research execution error: {e}")
            status_text.error(f"Research failed: {str(e)}")
            st.error(f"Research execution failed: {str(e)}")
            
            # Provide helpful suggestions based on error type
            if "provider" in str(e).lower() or "llm" in str(e).lower():
                st.info("💡 Try switching LLM providers or checking your configuration")
            elif "import" in str(e).lower():
                st.info("💡 Some dependencies may be missing. Check your environment setup.")
            elif "connection" in str(e).lower() or "network" in str(e).lower():
                st.info("💡 Check your network connection and LLM service availability")
            
            return
        
        
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
            
            # Add download buttons
            # Generate and offer PDF download
            pdf_bytes = markdown_to_pdf(results['markdown_report'], "generated_text")
            if pdf_bytes:
                # Check if it's actual PDF or HTML fallback
                if pdf_bytes.startswith(b'<!DOCTYPE html'):
                    # HTML fallback
                    st.download_button(
                        label="🌐 Download as HTML",
                        data=pdf_bytes,
                        file_name=f"research_report_{results.get('timestamp', 'unknown')}.html",
                        mime="text/html",
                        help="Download as HTML (PDF generation requires additional dependencies)"
                    )
                else:
                    # Actual PDF
                    st.download_button(
                        label="📕 Download as PDF",
                        data=pdf_bytes,
                        file_name=f"research_report_{results.get('timestamp', 'unknown')}.pdf",
                        mime="application/pdf",
                        help="Download the complete research report as a PDF file"
                    )
            
        elif results.get('generated_report'):
            # Display the generated report from the agent
            st.markdown(results['generated_report'])
            
            # Add download buttons
            # Generate and offer PDF download
            pdf_bytes = markdown_to_pdf(results['generated_report'], "Research Report")
            if pdf_bytes:
                # Check if it's actual PDF or HTML fallback
                if pdf_bytes.startswith(b'<!DOCTYPE html'):
                    # HTML fallback
                    st.download_button(
                        label="🌐 Download as HTML",
                        data=pdf_bytes,
                        file_name=f"research_report_{results.get('timestamp', 'unknown')}.html",
                        mime="text/html",
                        help="Download as HTML (PDF generation requires additional dependencies)"
                    )
                else:
                    # Actual PDF
                    st.download_button(
                        label="📕 Download as PDF",
                        data=pdf_bytes,
                        file_name=f"research_report_{results.get('timestamp', 'unknown')}.pdf",
                        mime="application/pdf",
                        help="Download the complete research report as a PDF file"
                    )
        
        elif results.get('generated_text'):
            # Fallback to displaying just the analysis if no full report
            st.markdown("---")
            st.markdown("## Research Analysis")
            st.markdown(results['generated_text'])
        elif results.get('error'):
            st.error(f"Error: {results['error']}")
        else:
            st.warning("Research analysis not available")
        
    except Exception as e:
        st.error(f"Failed to display results: {str(e)}")
        logger.error(f"Display results error: {e}")


def generate_report(results):
    """Generate markdown formatted report"""

    if results.get("generated_report"):
        return results["generated_report"]

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



if __name__ == "__main__":
    main()
