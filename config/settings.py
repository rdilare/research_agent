"""
Configuration loader and settings management
"""
import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file and environment variables"""
    try:
        # Load from YAML file
        config_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config = _override_with_env(config)
        
        logger.info("Configuration loaded successfully")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return _get_default_config()


def _override_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with environment variables"""
    
    # LLM settings
    if os.getenv('LLM_BASE_URL'):
        config['llm']['base_url'] = os.getenv('LLM_BASE_URL')
    
    if os.getenv('LLM_MODEL'):
        config['llm']['model'] = os.getenv('LLM_MODEL')
    
    return config


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration"""
    return {
        'app': {
            'name': 'Research Assistant Agent',
            'version': '1.0.0',
            'debug': False
        },
        'llm': {
            'provider': 'ollama',
            'model': 'llama3.2',
            'base_url': 'http://localhost:11434',
            'temperature': 0.7,
            'max_tokens': 2048
        },
        'vector_store': {
            'type': 'faiss',
            'embedding_model': 'all-MiniLM-L6-v2',
            'index_path': 'embeddings/faiss_index',
            'chunk_size': 512,
            'chunk_overlap': 50
        },
        'data_sources': {
            'arxiv': {'enabled': True, 'max_results': 10},
            'duckduckgo': {'enabled': True, 'max_results': 20, 'extract_full_content': True}
        },
        'cache': {
            'type': 'file',
            'ttl': 3600,
            'max_size': 1000
        },
        'reports': {
            'output_dir': 'reports/output',
            'template_dir': 'reports/templates',
            'formats': ['markdown', 'pdf']
        },
        'logging': {
            'level': 'INFO',
            'format': 'json',
            'file': 'logs/app.log',
            'debug_mode': False,
            'step_logging': True,
            'max_log_size': '10MB',
            'backup_count': 5
        }
    }


def setup_logging(config: Dict[str, Any]):
    """Setup logging based on configuration"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'ERROR')
    
    # Setup basic logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("Logging configured successfully")
