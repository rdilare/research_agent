"""
Research Assistant Agent - Main Entry Point
Standard LangChain/LangGraph Implementation
"""
import streamlit as st
from ui.app import main as streamlit_main
from config.settings import load_config, setup_logging
import logging


def main():
    """Main application entry point"""
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Research Assistant Agent with standard LangChain/LangGraph components")
        
        # Launch Streamlit interface
        streamlit_main()
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise


if __name__ == "__main__":
    main()
