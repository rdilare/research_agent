# Research Assistant Agent Architecture

## Project Overview
**Name:** Research Assistant Agent  
**Description:** LangGraph-based AI research system using standard LangChain components for comprehensive research across all domains. Built with LLaMA 3.2 via Ollama, retrieval-augmented generation, and structured report generation with citations.

**Goals:**
- Unified research across all domains using standard LangChain retrievers and tools
- LangGraph workflow orchestration with standard state management
- Standard RAG implementation with LangChain components

---


## Features
- Standard LangChain/LangGraph architecture and components
- Built-in ArxivRetriever for academic research
- DuckDuckGoSearchResults tool for web search
- Standard FAISS vector store with HuggingFace embeddings
- Ollama LLM integration via langchain-ollama
- Standard RAG chain implementation
- LangGraph StateGraph with TypedDict state management
- Structured report generation (Markdown format)
- Standard LangChain message passing and state management

---

## Software Architecture
### Standard LangChain/LangGraph Design Patterns
- **StateGraph Workflow:** Uses LangGraph's StateGraph with TypedDict state management
- **Standard Components:** ArxivRetriever, DuckDuckGoSearchResults, FAISS, HuggingFaceEmbeddings
- **Message Passing:** Standard LangChain message format (HumanMessage, AIMessage)
- **RAG Chain:** Standard LangChain RAG implementation with retrievers and chains
- **LLM Integration:** langchain-ollama for local LLaMA model integration
- **State Management:** TypedDict with annotated message passing

### Implementation Brief
- **LangGraph Workflow:**
  - StateGraph with standard node/edge definitions
  - TypedDict state with annotated message lists
  - Standard START/END node management
- **LangChain Components:**
  - ArxivRetriever for academic paper retrieval
  - DuckDuckGoSearchResults for web search
  - FAISS vectorstore with HuggingFace embeddings
  - Standard prompt templates and output parsers
- **RAG Implementation:**
  - Standard retriever.as_retriever() pattern
  - LangChain RunnablePassthrough for context flow
  - ChatPromptTemplate for structured prompts

---


## Technologies
- **LLM:** llama3.2 via langchain-ollama
- **Orchestration:** LangGraph StateGraph
- **Vector Store:** LangChain FAISS integration
- **Embeddings:** HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **Retrievers:** ArxivRetriever, FAISS retriever
- **Tools:** DuckDuckGoSearchResults
- **Prompts:** ChatPromptTemplate
- **Chains:** Standard RAG chain with RunnablePassthrough
- **State:** TypedDict with annotated message passing
- **Frontend:** Streamlit

---

## Components
### LangGraph StateGraph Workflow
- **Description:** Standard LangGraph StateGraph implementation
- **State Management:** TypedDict with annotated message lists
- **Nodes:** Standard LangChain component integration
- **Edges:** Sequential flow with START/END management

### Standard LangChain Components
- **ArxivRetriever:** Built-in academic paper retrieval
- **DuckDuckGoSearchResults:** Web search tool
- **FAISS Vectorstore:** Document similarity search
- **HuggingFaceEmbeddings:** Text embeddings
- **OllamaLLM:** Local LLM integration
- **ChatPromptTemplate:** Structured prompts
- **RAG Chain:** Standard retrieval-augmented generation

### Workflow Nodes
1. **Query Analysis Node**
   - Uses ChatPromptTemplate and OllamaLLM
   - Analyzes query to determine domain and strategy
   
2. **Document Retrieval Node**
   - ArxivRetriever for academic content
   - DuckDuckGoSearchResults for web content
   - Returns structured document list
   
3. **RAG Processing Node**
   - Creates FAISS vectorstore from documents
   - Standard retriever pattern
   - RAG chain with context formatting
   
4. **Report Generation Node**
   - Formats final markdown report
   - Includes sources and citations

### LangGraph Workflow
- **Type:** Standard StateGraph with TypedDict state
- **Description:** LangGraph workflow using standard LangChain components
- **Flow:**
  1. **analyze_query:** LLM-based query analysis using ChatPromptTemplate
  2. **retrieve_documents:** ArxivRetriever + DuckDuckGoSearchResults
  3. **process_with_rag:** FAISS vectorstore + standard RAG chain
  4. **generate_report:** Markdown report generation
- **State Structure:**
  ```python
  class AgentState(TypedDict):
      messages: Annotated[list, add_messages]
      original_query: str
      query_analysis: Dict[str, Any]
      raw_documents: List[Dict[str, Any]]
      retrieved_docs: List[str]
      generated_text: str
      markdown_report: str
      timestamp: str
      errors: List[str]
  ```
- **Implementation:**
  - Standard LangGraph node/edge definitions
  - Error handling with state error tracking
  - Execution logging and timing

---

## Configuration Management
- **Config Files:**
  - config/settings.yaml: main application settings
  - .env: environment variables (API keys, secrets, runtime options)
- **Secrets Handling:**
  - Use environment variables for sensitive data
  - Never commit secrets to version control
  - Optionally use secret managers (e.g., HashiCorp Vault, AWS Secrets Manager)
- **Runtime Config:**
  - Allow override of config via CLI args or environment variables
  - Validate config at startup and log missing/invalid settings

---

## Logging Guidelines
- Use structured logging (JSON or key-value pairs) for all workflow steps
- Log at appropriate levels: INFO (workflow progress), WARNING (recoverable issues), ERROR (failures)
- Include trace IDs/session IDs for request correlation
- Log API calls, data retrieval, LLM invocations, chart generation, and report compilation
- Store logs locally and/or forward to centralized logging (e.g., ELK, CloudWatch)
- Mask sensitive data in logs
- Monitor logs for error patterns and alert on critical failures

---

## Folder Structure
```
research-assistant/
├── README.md
├── requirements.txt
├── main.py
├── config/
│   └── settings.yaml
├── agents/
│   ├── graph.py                  # Standard LangGraph StateGraph
│   └── __init__.py
├── ui/
│   └── app.py                    # Streamlit interface
├── config/
│   └── settings.yaml
└── data/
    └── processed/
```

---

## Workflow Steps
1. **analyze_query** - LLM-based query analysis using ChatPromptTemplate
2. **retrieve_documents** - ArxivRetriever + DuckDuckGoSearchResults integration
3. **process_with_rag** - FAISS vectorstore creation and RAG chain execution
4. **generate_report** - Markdown report compilation with sources

---

## Data Flow
- User provides query via Streamlit interface
- LangGraph StateGraph executes workflow nodes
- ArxivRetriever fetches academic papers
- DuckDuckGoSearchResults fetches web content
- FAISS vectorstore enables similarity search
- Standard RAG chain generates analysis
- Markdown report returned to interface

---

## Standard LangChain Integration Points
- **ArxivRetriever:** Built-in academic paper retrieval
- **DuckDuckGoSearchResults:** Web search functionality
- **FAISS:** Vector similarity search
- **HuggingFaceEmbeddings:** Text embeddings
- **OllamaLLM:** Local LLaMA integration
- **ChatPromptTemplate:** Structured prompts
- **RunnablePassthrough:** Chain context flow
- **StateGraph:** LangGraph workflow orchestration

---

## Extensibility
- Add new LangChain retrievers (PubMedRetriever, WikipediaRetriever)
- Integrate additional LangChain tools
- Extend StateGraph with conditional routing
- Add new LangChain document loaders

---

## Future Extensions
- LangSmith for tracing and monitoring
- Additional LangChain community integrations
- Multi-agent LangGraph workflows
