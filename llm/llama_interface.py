"""
LLaMA Interface - Adapter for LLaMA model via Ollama
"""
from typing import List, Dict, Any, Optional
import logging
import requests
import json

logger = logging.getLogger(__name__)


class LlamaInterface:
    """Interface for LLaMA model via Ollama"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model', 'llama3.2:3b')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using LLaMA model
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            # Prepare messages
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text
            
        except requests.RequestException as e:
            logger.error(f"LLaMA API request failed: {e}")
            return f"Error: LLM request failed - {str(e)}"
        except Exception as e:
            logger.error(f"LLaMA generation failed: {e}")
            return f"Error: LLM generation failed - {str(e)}"
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured output following a schema
        
        Args:
            prompt: User prompt
            schema: Expected output schema
            
        Returns:
            Structured response dictionary
        """
        try:
            # Add schema instruction to prompt
            schema_prompt = f"""
            {prompt}
            
            Please respond in the following JSON format:
            {json.dumps(schema, indent=2)}
            
            Ensure your response is valid JSON that matches this schema exactly.
            """
            
            response = self.generate(schema_prompt)
            
            # Try to parse as JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"error": "Could not parse structured response", "raw_response": response}
            
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return {"error": str(e)}
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat interface with conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated response
        """
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
            
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return f"Error: Chat failed - {str(e)}"
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize long text"""
        prompt = f"""
        Please provide a concise summary of the following text in approximately {max_length} words:
        
        {text}
        
        Summary:
        """
        
        return self.generate(prompt)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        prompt = f"""
        Analyze the sentiment of the following text and respond in JSON format:
        
        Text: {text}
        
        Respond with:
        {{
            "sentiment": "positive/negative/neutral",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """
        
        return self.generate_structured(prompt, {
            "sentiment": "string",
            "confidence": "number",
            "reasoning": "string"
        })
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        prompt = f"""
        Extract the main entities (people, organizations, locations, concepts) from the following text.
        Return only a JSON list of entities.
        
        Text: {text}
        
        Entities:
        """
        
        response = self.generate(prompt)
        try:
            # Try to parse as JSON list
            import re
            list_match = re.search(r'\[.*\]', response, re.DOTALL)
            if list_match:
                return json.loads(list_match.group())
            else:
                # Fallback: split by lines/commas
                entities = [e.strip() for e in response.split('\n') if e.strip()]
                return entities[:10]  # Limit to top 10
        except:
            return []
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
