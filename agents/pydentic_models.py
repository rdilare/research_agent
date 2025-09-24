"""
Pydantic models for structured LLM outputs in the research agent
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ReportSection(BaseModel):
    """Model for report sections containing title and sub-queries"""
    title: str = Field(..., description="Title of the report section")
    sub_queries: List[str] = Field(..., description="List of focused sub-queries for this section")


class ReportPlan(BaseModel):
    """Model for the complete report planning structure"""
    report_sections: List[ReportSection] = Field(..., description="List of report sections with sub-queries")
    report_title: str = Field(..., description="Title of the research report")


class SectionContent(BaseModel):
    """Model for generated section content"""
    section_title: str = Field(default="", description="Title of the section")
    content: str = Field(default="", description="Generated content for the section") 
    sources_used: Optional[List[str]] = Field(default_factory=list, description="List of sources or queries used")


class ResearchAnalysis(BaseModel):
    """Model for query analysis results"""
    research_type: str = Field(..., description="Type of research (e.g., academic, market, technical)")
    complexity: str = Field(..., description="Complexity level (simple, moderate, complex)")
    key_topics: List[str] = Field(..., description="Main topics to be covered")
    suggested_sources: List[str] = Field(..., description="Recommended source types")


class WebSearchResult(BaseModel):
    """Model for structured web search results"""
    title: str = Field(..., description="Title of the search result")
    snippet: str = Field(..., description="Content snippet from the result")
    url: Optional[str] = Field(default=None, description="URL of the source")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score (0-1)")


class ReportGeneration(BaseModel):
    """Model for complete report generation output"""
    title: str = Field(..., description="Report title")
    executive_summary: Optional[str] = Field(default=None, description="Executive summary")
    sections: List[SectionContent] = Field(..., description="Generated report sections")
    conclusions: Optional[str] = Field(default=None, description="Key conclusions")
    references: Optional[List[str]] = Field(default_factory=list, description="List of references used")