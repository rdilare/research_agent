"""
Citation Manager - Manages and formats citations for research sources
"""
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CitationManager:
    """Tool for managing and formatting citations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.citations = []
        self.formatters = {
            'markdown': self._format_markdown,
            'bibtex': self._format_bibtex,
            'apa': self._format_apa,
            'mla': self._format_mla
        }
    
    def add_citation(self, source_data: Dict[str, Any]) -> str:
        """
        Add a citation from source data
        
        Args:
            source_data: Dictionary containing source information
            
        Returns:
            Citation ID for reference
        """
        try:
            citation_id = f"ref_{len(self.citations) + 1}"
            
            citation = {
                'id': citation_id,
                'title': source_data.get('title', 'Unknown Title'),
                'authors': source_data.get('authors', []),
                'url': source_data.get('url', ''),
                'source': source_data.get('source', 'unknown'),
                'publication_date': source_data.get('publication_date', ''),
                'access_date': datetime.now().strftime('%Y-%m-%d'),
                'doc_type': source_data.get('doc_type', 'webpage'),
                'abstract': source_data.get('abstract', ''),
                'doi': source_data.get('doi', ''),
                'journal': source_data.get('journal', ''),
                'volume': source_data.get('volume', ''),
                'issue': source_data.get('issue', ''),
                'pages': source_data.get('pages', '')
            }
            
            self.citations.append(citation)
            logger.debug(f"Added citation: {citation_id}")
            
            return citation_id
            
        except Exception as e:
            logger.error(f"Failed to add citation: {e}")
            return ""
    
    def format_citations(self, format_type: str = 'markdown') -> str:
        """
        Format all citations in specified format
        
        Args:
            format_type: Format type (markdown, bibtex, apa, mla)
            
        Returns:
            Formatted citations string
        """
        try:
            if format_type not in self.formatters:
                format_type = 'markdown'
            
            formatter = self.formatters[format_type]
            formatted_citations = []
            
            for citation in self.citations:
                formatted = formatter(citation)
                if formatted:
                    formatted_citations.append(formatted)
            
            if format_type == 'bibtex':
                return '\n\n'.join(formatted_citations)
            else:
                return '\n'.join(formatted_citations)
                
        except Exception as e:
            logger.error(f"Citation formatting failed: {e}")
            return ""
    
    def get_citation_by_id(self, citation_id: str) -> Dict[str, Any]:
        """Get citation data by ID"""
        for citation in self.citations:
            if citation['id'] == citation_id:
                return citation
        return {}
    
    def _format_markdown(self, citation: Dict[str, Any]) -> str:
        """Format citation in Markdown"""
        try:
            authors_str = self._format_authors(citation['authors'])
            title = citation['title']
            url = citation['url']
            access_date = citation['access_date']
            
            if citation['doc_type'] == 'academic_paper':
                # Academic paper format
                pub_date = citation.get('publication_date', '')
                journal = citation.get('journal', '')
                
                if journal:
                    return f"- {authors_str} ({pub_date}). **{title}**. *{journal}*. [{url}]({url}) (accessed {access_date})"
                else:
                    return f"- {authors_str} ({pub_date}). **{title}**. [{url}]({url}) (accessed {access_date})"
            else:
                # Web page format
                source = citation.get('source', '').title()
                return f"- {authors_str} **{title}**. *{source}*. [{url}]({url}) (accessed {access_date})"
                
        except Exception as e:
            logger.error(f"Markdown formatting failed: {e}")
            return ""
    
    def _format_bibtex(self, citation: Dict[str, Any]) -> str:
        """Format citation in BibTeX"""
        try:
            citation_id = citation['id']
            title = citation['title']
            authors = ' and '.join(citation['authors']) if citation['authors'] else 'Unknown'
            url = citation['url']
            year = self._extract_year(citation.get('publication_date', ''))
            
            if citation['doc_type'] == 'academic_paper':
                entry_type = 'article'
                journal = citation.get('journal', '')
                volume = citation.get('volume', '')
                pages = citation.get('pages', '')
                
                bibtex = f"@{entry_type}{{{citation_id},\n"
                bibtex += f"  title = {{{title}}},\n"
                bibtex += f"  author = {{{authors}}},\n"
                if year:
                    bibtex += f"  year = {{{year}}},\n"
                if journal:
                    bibtex += f"  journal = {{{journal}}},\n"
                if volume:
                    bibtex += f"  volume = {{{volume}}},\n"
                if pages:
                    bibtex += f"  pages = {{{pages}}},\n"
                bibtex += f"  url = {{{url}}}\n}}"
                
            else:
                entry_type = 'misc'
                bibtex = f"@{entry_type}{{{citation_id},\n"
                bibtex += f"  title = {{{title}}},\n"
                bibtex += f"  author = {{{authors}}},\n"
                if year:
                    bibtex += f"  year = {{{year}}},\n"
                bibtex += f"  url = {{{url}}},\n"
                bibtex += f"  note = {{Accessed: {citation['access_date']}}}\n}}"
            
            return bibtex
            
        except Exception as e:
            logger.error(f"BibTeX formatting failed: {e}")
            return ""
    
    def _format_apa(self, citation: Dict[str, Any]) -> str:
        """Format citation in APA style"""
        try:
            authors_str = self._format_authors_apa(citation['authors'])
            title = citation['title']
            url = citation['url']
            year = self._extract_year(citation.get('publication_date', ''))
            
            if citation['doc_type'] == 'academic_paper':
                journal = citation.get('journal', '')
                if journal:
                    return f"{authors_str} ({year}). {title}. *{journal}*. Retrieved from {url}"
                else:
                    return f"{authors_str} ({year}). {title}. Retrieved from {url}"
            else:
                return f"{authors_str} ({year}). {title}. Retrieved from {url}"
                
        except Exception as e:
            logger.error(f"APA formatting failed: {e}")
            return ""
    
    def _format_mla(self, citation: Dict[str, Any]) -> str:
        """Format citation in MLA style"""
        try:
            authors_str = self._format_authors_mla(citation['authors'])
            title = citation['title']
            url = citation['url']
            access_date = citation['access_date']
            
            return f"{authors_str} \"{title}.\" Web. {access_date}. <{url}>"
                
        except Exception as e:
            logger.error(f"MLA formatting failed: {e}")
            return ""
    
    def _format_authors(self, authors: List[str]) -> str:
        """Format authors list for citations"""
        if not authors:
            return "Unknown Author"
        elif len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        else:
            return f"{authors[0]} et al."
    
    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors for APA style"""
        if not authors:
            return "Unknown Author"
        
        formatted_authors = []
        for author in authors:
            # Convert "First Last" to "Last, F."
            parts = author.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                first_initial = parts[0][0] if parts[0] else ""
                formatted_authors.append(f"{last_name}, {first_initial}.")
            else:
                formatted_authors.append(author)
        
        if len(formatted_authors) == 1:
            return formatted_authors[0]
        elif len(formatted_authors) == 2:
            return f"{formatted_authors[0]}, & {formatted_authors[1]}"
        else:
            return ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
    
    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors for MLA style"""
        if not authors:
            return "Unknown Author."
        elif len(authors) == 1:
            return f"{authors[0]}."
        else:
            return f"{authors[0]}, et al."
    
    def _extract_year(self, date_str: str) -> str:
        """Extract year from date string"""
        if not date_str:
            return ""
        
        # Try to extract 4-digit year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        return year_match.group() if year_match else ""
    
    def export_to_zotero(self) -> bool:
        """Export citations to Zotero (placeholder implementation)"""
        # This would implement Zotero API integration
        logger.info("Zotero export not yet implemented")
        return False
    
    def clear_citations(self):
        """Clear all citations"""
        self.citations = []
        logger.info("Citations cleared")
    
    def get_citation_count(self) -> int:
        """Get total number of citations"""
        return len(self.citations)
