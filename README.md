# Research Assistant Agent

## Standard LangChain/LangGraph Implementation

A research assistant built using ** LangChain and LangGraph ** for comprehensive research across domains.

### Features

- **Standard LangGraph StateGraph** workflow orchestration
- **ArxivRetriever** for academic paper retrieval
- **DuckDuckGoSearchResults** for web search
- **FAISS vectorstore** with HuggingFace embeddings
- **Ollama LLM** integration via langchain-ollama
- **Standard RAG chain** implementation
- **Streamlit** web interface

### Architecture

The system uses standard LangChain/LangGraph patterns:

1. **StateGraph Workflow** - LangGraph StateGraph with TypedDict state management
2. **Standard Components** - Built-in LangChain retrievers and tools
3. **RAG Chain** - Standard retrieval-augmented generation
4. **Message Passing** - LangChain message format (HumanMessage, AIMessage)

### Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

### Workflow

1. **analyze_query** - LLM-based query analysis
2. **retrieve_documents** - ArxivRetriever + DuckDuckGoSearchResults
3. **process_with_rag** - FAISS vectorstore + standard RAG chain
4. **generate_report** - Markdown report generation


### Configuration

Edit `config/settings.yaml` to customize:

```yaml
llm:
  model: "llama3.2"
  base_url: "http://localhost:11434"

embeddings:
  model: "all-MiniLM-L6-v2"

retrieval:
  max_results: 5
```
