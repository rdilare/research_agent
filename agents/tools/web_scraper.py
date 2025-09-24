"""
Web Scraper Tool - Extracts data from websites and online resources
"""
from typing import Dict, Any, List, Optional
import logging
import requests
from bs4 import BeautifulSoup
import time
import re

logger = logging.getLogger(__name__)


class WebScraper:
    """Tool for web scraping and content extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.request_delay = 1  # Delay between requests in seconds
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a single URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with scraped content and metadata
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            content = self._extract_content(soup)
            
            # Extract metadata
            metadata = self._extract_metadata(soup, url)
            
            result = {
                'url': url,
                'title': metadata.get('title', ''),
                'content': content,
                'metadata': metadata,
                'status': 'success'
            }
            
            logger.info(f"Successfully scraped: {url}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {'url': url, 'status': 'error', 'error': str(e)}
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            return {'url': url, 'status': 'error', 'error': str(e)}
    
    def scrape_entity_data(self, entity: str) -> Dict[str, Any]:
        """
        Scrape data related to a specific entity
        
        Args:
            entity: Entity name to search for
            
        Returns:
            Dictionary with scraped data about the entity
        """
        try:
            # Simple search on a few reliable sources
            search_urls = self._generate_search_urls(entity)
            
            results = []
            for url in search_urls[:3]:  # Limit to first 3 URLs
                scraped_data = self.scrape_url(url)
                if scraped_data.get('status') == 'success':
                    results.append(scraped_data)
                
                # Rate limiting
                time.sleep(self.request_delay)
            
            return {
                'entity': entity,
                'results': results,
                'total_sources': len(results)
            }
            
        except Exception as e:
            logger.error(f"Entity scraping failed for {entity}: {e}")
            return {'entity': entity, 'error': str(e)}
    
    def scrape_structured_data(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """
        Scrape structured data using CSS selectors
        
        Args:
            url: URL to scrape
            selectors: Dictionary mapping field names to CSS selectors
            
        Returns:
            Dictionary with extracted structured data
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            structured_data = {}
            for field, selector in selectors.items():
                try:
                    elements = soup.select(selector)
                    if elements:
                        structured_data[field] = [elem.get_text(strip=True) for elem in elements]
                    else:
                        structured_data[field] = []
                except Exception as e:
                    logger.warning(f"Failed to extract {field} using selector {selector}: {e}")
                    structured_data[field] = []
            
            return {
                'url': url,
                'data': structured_data,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Structured scraping failed for {url}: {e}")
            return {'url': url, 'status': 'error', 'error': str(e)}
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from parsed HTML"""
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.post-content', '.entry-content', '.article-body'
        ]
        
        content_text = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content_text = content_elem.get_text(separator=' ', strip=True)
                break
        
        # Fallback to body content
        if not content_text:
            body = soup.find('body')
            if body:
                content_text = body.get_text(separator=' ', strip=True)
        
        # Clean up text
        content_text = re.sub(r'\s+', ' ', content_text)
        return content_text[:5000]  # Limit content length
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from parsed HTML"""
        metadata = {'url': url}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Open Graph data
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            metadata['og_description'] = og_desc.get('content', '')
        
        # Publication date
        date_selectors = [
            'meta[name="article:published_time"]',
            'meta[property="article:published_time"]',
            'time[datetime]',
            '.published-date',
            '.post-date'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_value = date_elem.get('content') or date_elem.get('datetime') or date_elem.get_text(strip=True)
                if date_value:
                    metadata['published_date'] = date_value
                    break
        
        return metadata
    
    def _generate_search_urls(self, entity: str) -> List[str]:
        """Generate search URLs for an entity"""
        # This is a simplified version - in practice, you'd use proper search APIs
        search_queries = [
            f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}",
            # Add more reliable sources as needed
        ]
        
        return search_queries
    
    def extract_links(self, url: str, filter_pattern: Optional[str] = None) -> List[str]:
        """Extract links from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                
                # Filter links if pattern provided
                if filter_pattern:
                    if re.search(filter_pattern, href):
                        links.append(href)
                else:
                    if href.startswith('http'):
                        links.append(href)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Link extraction failed for {url}: {e}")
            return []
