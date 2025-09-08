"""
LangGraph Agent Workflow Orchestration using Standard LangChain/LangGraph Components
"""
from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # Use community Ollama instead
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Standard LangGraph state definition using TypedDict"""
    messages: Annotated[list, add_messages]
    original_query: str
    query_analysis: Dict[str, Any]
    raw_documents: List[Dict[str, Any]]
    retrieved_docs: List[str]
    generated_text: str
    markdown_report: str
    timestamp: str
    errors: List[str]
    
    # Execution tracking
    step_logs: List[Dict[str, Any]]
    execution_times: Dict[str, float]


class ResearchAgentGraph:
    """Standard LangGraph-based research agent workflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize standard LangChain components
        self.llm = Ollama(
            model=config.get('llm', {}).get('model', 'llama3.2'),
            base_url=config.get('llm', {}).get('base_url', 'http://localhost:11434')
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.get('embeddings', {}).get('model', 'all-MiniLM-L6-v2')
        )
        
        # Standard LangChain retrievers and tools
        self.arxiv_retriever = ArxivRetriever(load_max_docs=5)
        self.web_search_tool = DuckDuckGoSearchResults(max_results=5, output_format="list")
        
        # Initialize vector store (will be populated during retrieval)
        self.vector_store = None
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze user query using standard LangChain prompt template"""
        start_time = time.time()
        
        try:
            query = state["messages"][-1].content if state["messages"] else state.get("original_query", "")
            
            # Standard LangChain prompt for query analysis
            analysis_prompt = ChatPromptTemplate.from_template("""
            Analyze this research query and determine:
            1. Research domain (academic, market, technology, general)
            2. Key entities and topics
            3. Required data sources: "web", or "academic" or both.
            4. Search strategy
            
            Query: {query}
            
            Respond only with a JSON object containing: domain, entities, topics, sources, strategy
            like:
            {{
                    "domain": "general",
                    "entities": ['the user query'],
                    "topics": ['dog,' 'cat', ],
                    "sources": ["web", "academic"],
                    "strategy": "broad search"
                }}
            """)
            
            chain = analysis_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})

            dec = "\n"+("="*10)+"\n"
            print(dec, result, dec)
            
            # Parse the analysis result
            try:
                analysis = json.loads(result)
            except:
                analysis = {
                    "domain": "general",
                    "entities": [query],
                    "topics": [query],
                    "sources": ["web", "academic"],
                    "strategy": "broad search"
                }
            
            state["query_analysis"] = analysis
            state["original_query"] = query
            
            execution_time = time.time() - start_time
            self._log_step("analyze_query", query, analysis, execution_time, state)
            
        except Exception as e:
            error_msg = f"Query analysis failed: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["query_analysis"] = {"domain": "general", "entities": [], "topics": [], "sources": ["web"]}
        
        return state
    
    def retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve documents using standard LangChain retrievers"""
        start_time = time.time()
        
        try:
            query = state["original_query"]
            analysis = state["query_analysis"]
            documents = []
            
            # Use ArxivRetriever for academic queries
            if False:
            # if "academic" in analysis.get("sources", []) or analysis.get("domain") == "academic":
                try:
                    arxiv_docs = self.arxiv_retriever.get_relevant_documents(query)
                    for doc in arxiv_docs[:3]:  # Limit results
                        documents.append({
                            "content": doc.page_content,
                            "source": "arxiv",
                            "metadata": doc.metadata,
                            "title": doc.metadata.get("Title", ""),
                            "url": doc.metadata.get("entry_id", "")
                        })
                except Exception as e:
                    logger.warning(f"Arxiv retrieval failed: {e}")
            
            # Use web search tool for general/market queries
            if "web" in analysis.get("sources", []) or analysis.get("domain") in ["market", "general"]:
                try:
                    web_results = self.web_search_tool.invoke(query)
                    print("\n"+"="*10)
                    print(type(web_results))
                    print(web_results)
                    print("\n"+"="*10)
                    if isinstance(web_results, list):
                        for result in web_results[:3]:  # Limit results
                            documents.append({
                                "content": result.get("snippet", ""),
                                "source": "web",
                                "metadata": result,
                                "title": result.get("title", ""),
                                "url": result.get("link", "")
                            })
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
            
            state["raw_documents"] = documents
            
            execution_time = time.time() - start_time
            self._log_step("retrieve_documents", query, f"{len(documents)} documents", execution_time, state)
            
        except Exception as e:
            error_msg = f"Document retrieval failed: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["raw_documents"] = []
        
        return state
    
    def process_with_rag(self, state: AgentState) -> AgentState:
        """Process documents using standard LangChain RAG chain"""
        start_time = time.time()
        
        try:
            documents = state["raw_documents"]
            query = state["original_query"]
            
            if not documents:
                state["retrieved_docs"] = []
                state["generated_text"] = "No documents found for analysis."
                return state
            
            # Create vector store from documents
            doc_texts = [doc["content"] for doc in documents if doc.get("content")]
            if doc_texts:
                self.vector_store = FAISS.from_texts(doc_texts, self.embeddings)
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # Standard RAG chain
                rag_prompt = ChatPromptTemplate.from_template("""
                Based on the following context, provide a comprehensive analysis for the query.
                
                Context: {context}
                
                Query: {question}
                
                Provide a detailed analysis with key insights and findings.
                """)
                
                rag_chain = (
                    {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
                    | rag_prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(query)
                state["generated_text"] = response
                state["retrieved_docs"] = doc_texts[:3]
            else:
                state["generated_text"] = "No valid content found in retrieved documents."
                state["retrieved_docs"] = []
            
            execution_time = time.time() - start_time
            self._log_step("process_with_rag", f"{len(doc_texts)} docs", "Generated response", execution_time, state)
            
        except Exception as e:
            error_msg = f"RAG processing failed: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["generated_text"] = "Analysis could not be completed due to processing errors."
            state["retrieved_docs"] = []
        
        return state
    
    def generate_report(self, state: AgentState) -> AgentState:
        """Generate final markdown report"""
        start_time = time.time()
        
        try:
            query = state["original_query"]
            analysis = state["generated_text"]
            documents = state["raw_documents"]
            
            # Generate markdown report
            report_parts = [
                f"# Research Report: {query}",
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Analysis",
                analysis,
                "",
                "## Sources",
            ]
            
            # Add sources
            for i, doc in enumerate(documents, 1):
                title = doc.get("title", "Unknown Title")
                url = doc.get("url", "")
                source = doc.get("source", "unknown")
                
                if url:
                    report_parts.append(f"{i}. [{title}]({url}) - *{source}*")
                else:
                    report_parts.append(f"{i}. {title} - *{source}*")
            
            markdown_report = "\n".join(report_parts)
            state["markdown_report"] = markdown_report
            state["timestamp"] = datetime.now().isoformat()
            
            execution_time = time.time() - start_time
            self._log_step("generate_report", "Report generation", "Markdown report", execution_time, state)
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["markdown_report"] = f"# Research Report\n\nError generating report: {str(e)}"
        
        return state
    
    def _format_docs(self, docs):
        """Format retrieved documents for RAG context"""
        return "\n\n".join(doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs)
    
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph using standard LangGraph patterns"""
        
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes using standard LangGraph approach
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("process_with_rag", self.process_with_rag)
        workflow.add_node("generate_report", self.generate_report)
        
        # Define the flow using standard LangGraph edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_edge("analyze_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "process_with_rag")
        workflow.add_edge("process_with_rag", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def _log_step(self, step_name: str, inputs: Any, outputs: Any, execution_time: float, state: Dict[str, Any]):
        """Log step execution details"""
        step_log = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'status': 'success' if not state.get('errors') else 'error',
            'input_summary': str(inputs)[:200],
            'output_summary': str(outputs)[:200],
        }
        
        if 'step_logs' not in state:
            state['step_logs'] = []
        if 'execution_times' not in state:
            state['execution_times'] = {}
            
        state['step_logs'].append(step_log)
        state['execution_times'][step_name] = execution_time
        
        logger.info(f"Step '{step_name}' completed in {execution_time:.2f}s")
    
    def run(self, query: str) -> Dict[str, Any]:
        """Execute the research workflow with standard LangGraph approach"""
        try:
            # Initialize state with standard LangGraph message format
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "query_analysis": {},
                "raw_documents": [],
                "retrieved_docs": [],
                "generated_text": "",
                "markdown_report": "",
                "timestamp": "",
                "errors": [],
                "step_logs": [],
                "execution_times": {}
            }
            
            # Execute the graph
            result = self.graph.invoke(initial_state)

            with open("debug_log", "w") as f:
                f.write(str(result))
                # f.write(json.dumps(result, indent=2))
            
            logger.info(f"Research workflow completed for query: {query}")
            return result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            return {
                "original_query": query,
                "generated_text": f"Research failed: {str(e)}",
                "markdown_report": f"# Research Failed\n\nError: {str(e)}",
                "errors": [error_msg],
                "timestamp": datetime.now().isoformat(),
                "step_logs": [],
                "execution_times": {}
            }
