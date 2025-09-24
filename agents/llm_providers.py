"""
LLM Provider Abstraction Layer

This module provides a unified interface for different LLM providers,
making it easy to switch between Ollama, OpenAI, Anthropic, etc.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model', 'default')
        self.temperature = config.get('temperature', 0.7)
        
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with a prompt and return the response"""
        pass
    
    @abstractmethod
    def predict(self, prompt: str) -> str:
        """Predict method for backward compatibility"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the LLM provider"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM Provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Try different import patterns for Ollama
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            from langchain_community.llms import Ollama
        
        self.llm = Ollama(
            model=self.model_name,
            base_url=config.get('base_url', 'http://localhost:11434'),
            temperature=self.temperature
        )
        
    def invoke(self, prompt: str) -> str:
        """Invoke Ollama LLM"""
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                return self.llm(prompt)
        except Exception as e:
            logger.error(f"Ollama invoke failed: {e}")
            raise
    
    def predict(self, prompt: str) -> str:
        """Predict method for backward compatibility"""
        if hasattr(self.llm, 'predict'):
            return self.llm.predict(prompt)
        return self.invoke(prompt)
    
    @property
    def provider_name(self) -> str:
        return "ollama"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider implementation (ready for future use)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Future implementation for OpenAI
        self.api_key = config.get('api_key')
        self.organization = config.get('organization')
        
    def invoke(self, prompt: str) -> str:
        """Invoke OpenAI LLM - placeholder implementation"""
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    def predict(self, prompt: str) -> str:
        """Predict method for backward compatibility"""
        return self.invoke(prompt)
    
    @property
    def provider_name(self) -> str:
        return "openai"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM Provider implementation (ready for future use)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key')
        
    def invoke(self, prompt: str) -> str:
        """Invoke Anthropic LLM - placeholder implementation"""
        raise NotImplementedError("Anthropic provider not yet implemented")
    
    def predict(self, prompt: str) -> str:
        """Predict method for backward compatibility"""
        return self.invoke(prompt)
    
    @property
    def provider_name(self) -> str:
        return "anthropic"


class LLMProviderFactory:
    """Factory class for creating LLM providers"""
    
    _providers = {
        'ollama': OllamaProvider,
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> BaseLLMProvider:
        """Create an LLM provider instance"""
        provider_type = provider_type.lower()
        
        if provider_type not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(f"Unknown provider '{provider_type}'. Available: {available}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new LLM provider"""
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError("Provider class must inherit from BaseLLMProvider")
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available provider names"""
        return list(cls._providers.keys())


def create_llm_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    """
    Convenience function to create an LLM provider from config
    
    Args:
        config: Configuration dictionary with 'provider' key and provider-specific config
        
    Returns:
        BaseLLMProvider instance
    """
    provider_type = config.get('provider', 'ollama')
    return LLMProviderFactory.create_provider(provider_type, config)