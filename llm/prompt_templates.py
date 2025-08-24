"""
Prompt Templates for LLM Interactions
"""

# Query Analysis Templates
QUERY_ANALYSIS_TEMPLATE = """
Analyze the following research query and extract key information:

Query: "{query}"

Please provide a structured analysis with:
1. Research Domain (academic, market, technology, general, etc.)
2. Key Entities and Concepts
3. Research Intent (overview, comparison, analysis, trends, etc.)
4. Main Topics/Themes
5. Suggested Data Sources

Format your response as:
Domain: [domain]
Entities: [entity1, entity2, ...]
Intent: [intent]
Topics: [topic1, topic2, ...]
Sources: [source1, source2, ...]
"""

# Research Synthesis Templates
SYNTHESIS_TEMPLATE = """
Based on the following research context, provide a comprehensive analysis:

Research Question: {query}

Context Information:
{context}

Please provide:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Analysis and Insights
4. Conclusions and Implications

Structure your response clearly with headers and maintain academic rigor.
"""

# SWOT Analysis Template
SWOT_TEMPLATE = """
Based on the research context provided, generate a SWOT analysis for: {subject}

Context:
{context}

Create a comprehensive SWOT analysis with:

STRENGTHS:
- [List internal positive factors]

WEAKNESSES:
- [List internal negative factors]

OPPORTUNITIES:
- [List external positive factors]

THREATS:
- [List external negative factors]

Provide specific, actionable insights based on the research data.
"""

# Market Analysis Template
MARKET_ANALYSIS_TEMPLATE = """
Analyze the market information for: {subject}

Research Data:
{context}

Provide a market analysis covering:

1. Market Overview
2. Key Players and Competition
3. Market Trends and Drivers
4. Challenges and Barriers
5. Growth Opportunities
6. Financial Insights (if available)

Use data from the research context to support your analysis.
"""

# Academic Summary Template
ACADEMIC_SUMMARY_TEMPLATE = """
Summarize the academic research on: {topic}

Academic Sources:
{context}

Provide an academic literature review covering:

1. Current State of Research
2. Key Findings and Methodologies
3. Research Gaps and Limitations
4. Future Research Directions
5. Practical Implications

Maintain academic tone and cite key insights from the provided sources.
"""

# Report Generation Template
REPORT_TEMPLATE = """
Generate a comprehensive research report on: {topic}

Research Context:
{context}

Tool Results:
{tool_results}

Charts and Visualizations:
{charts}

Create a structured report with:

# Executive Summary

# Introduction

# Methodology

# Key Findings

# Analysis and Discussion

# Visualizations and Data

# Conclusions and Recommendations

# Limitations

Ensure the report is professional, well-structured, and data-driven.
"""

# Data Validation Template
DATA_VALIDATION_TEMPLATE = """
Evaluate the quality and reliability of the following research data:

Data Sources:
{sources}

Content Sample:
{content_sample}

Assess:
1. Source Credibility
2. Data Completeness
3. Information Accuracy
4. Potential Biases
5. Overall Reliability Score (1-10)

Provide specific recommendations for improving data quality.
"""
