"""
Market Search Tool - Searches market data and news sources
"""
from typing import Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)


class MarketSearch:
    """Tool for searching market data, news, and trends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def search_duckduckgo(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo via ddgs with full content extraction
        
        Args:
            query: Search query
            analysis: Query analysis results
            
        Returns:
            List of search results with full content
        """
        try:
            # Import ddgs directly
            from ddgs import DDGS
            
            ddg_config = self.config.get('data_sources', {}).get('duckduckgo', {})
            max_results = ddg_config.get('max_results', 20)
            region = ddg_config.get('region', 'us-en')
            safesearch = ddg_config.get('safesearch', 'moderate')
            extract_content = ddg_config.get('extract_full_content', True)
            
            logger.info(f"DuckDuckGo search executed for query: '{query}' (extract_content: {extract_content})")
            
            # Perform search using ddgs
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results
                ))
            
            logger.debug(f"DuckDuckGo returned {len(search_results)} raw results")
            
            # Extract full content from each result if enabled
            if extract_content:
                results = self._process_search_results_with_content(search_results, query)
            else:
                results = self._process_search_results_basic(search_results, query)
            
            logger.info(f"Found {len(results)} results from DuckDuckGo")
            return results
            
        except ImportError:
            logger.error("ddgs library not available. Install with: pip install ddgs")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def search_bing(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search using Bing Search API (DISABLED)
        
        Args:
            query: Search query
            analysis: Query analysis results
            
        Returns:
            Empty list (feature disabled)
        """
        logger.info("Bing search is disabled")
        return []
    
    def search_google_trends(self, entities: List[str]) -> Dict[str, Any]:
        """
        Search Google Trends for entity popularity
        
        Args:
            entities: List of entities to check trends for
            
        Returns:
            Dictionary with trend data
        """
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            
            trends_data = {}
            
            # Process entities in batches (Google Trends has limits)
            batch_size = 5
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                
                try:
                    pytrends.build_payload(batch, cat=0, timeframe='today 3-m', geo='', gprop='')
                    
                    # Get interest over time
                    interest_data = pytrends.interest_over_time()
                    
                    for entity in batch:
                        if entity in interest_data.columns:
                            trends_data[entity] = {
                                'interest_over_time': interest_data[entity].tolist(),
                                'average_interest': interest_data[entity].mean()
                            }
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to get trends for batch {batch}: {e}")
                    continue
            
            logger.info(f"Retrieved trends data for {len(trends_data)} entities")
            return trends_data
            
        except ImportError:
            logger.error("pytrends library not available")
            return {}
        except Exception as e:
            logger.error(f"Google Trends search failed: {e}")
            return {}
    
    def _process_search_results_with_content(self, search_results: List[Dict], query: str) -> List[Dict[str, Any]]:
        """
        Process search results and extract full content from web pages
        
        Args:
            search_results: Raw search results from ddgs
            query: Original search query
            
        Returns:
            List of processed results with full content
        """
        processed_results = []
        
        for i, result in enumerate(search_results):
            try:
                # Basic result structure from ddgs
                basic_result = {
                    'source': 'duckduckgo',
                    'doc_type': 'web_page',
                    'title': result.get('title', ''),
                    'content': result.get('body', ''),  # ddgs uses 'body' instead of 'snippet'
                    'url': result.get('href', ''),
                    'relevance_score': 1.0 - (i * 0.05),  # Decrease relevance with position
                    'search_query': query,
                    'extraction_status': 'snippet_only'
                }
                
                # Try to extract full content from the webpage
                if basic_result['url']:
                    ddg_config = self.config.get('data_sources', {}).get('duckduckgo', {})
                    timeout = ddg_config.get('content_timeout', 10)
                    max_length = ddg_config.get('max_content_length', 5000)
                    
                    full_content = self._extract_webpage_content(basic_result['url'], timeout)
                    if full_content:
                        basic_result['content'] = full_content[:max_length]  # Limit content size
                        basic_result['extraction_status'] = 'full_content'
                        logger.debug(f"Extracted full content from {basic_result['url'][:50]}...")
                    else:
                        # Keep the snippet as fallback
                        basic_result['extraction_status'] = 'extraction_failed'
                        logger.debug(f"Failed to extract content from {basic_result['url'][:50]}, using snippet")
                
                # Only add if we have meaningful content
                if basic_result['title'] or (basic_result['content'] and len(basic_result['content'].strip()) > 50):
                    processed_results.append(basic_result)
                    logger.debug(f"Added result: {basic_result['title'][:50]}... ({len(basic_result['content'])} chars)")
                
            except Exception as e:
                logger.warning(f"Failed to process search result {i}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_results)} results with content extraction")
        return processed_results
    
    def _process_search_results_basic(self, search_results: List[Dict], query: str) -> List[Dict[str, Any]]:
        """
        Process search results using only snippets (no content extraction)
        
        Args:
            search_results: Raw search results from ddgs
            query: Original search query
            
        Returns:
            List of processed results with snippets only
        """
        processed_results = []
        
        for i, result in enumerate(search_results):
            try:
                # Basic result structure from ddgs
                basic_result = {
                    'source': 'duckduckgo',
                    'doc_type': 'web_page',
                    'title': result.get('title', ''),
                    'content': result.get('body', ''),  # ddgs uses 'body' instead of 'snippet'
                    'url': result.get('href', ''),
                    'relevance_score': 1.0 - (i * 0.05),  # Decrease relevance with position
                    'search_query': query,
                    'extraction_status': 'snippet_only'
                }
                
                # Only add if we have meaningful content
                if basic_result['title'] or (basic_result['content'] and len(basic_result['content'].strip()) > 10):
                    processed_results.append(basic_result)
                    logger.debug(f"Added result: {basic_result['title'][:50]}... ({len(basic_result['content'])} chars)")
                
            except Exception as e:
                logger.warning(f"Failed to process search result {i}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_results)} results (snippet only)")
        return processed_results
    
    def _extract_webpage_content(self, url: str, timeout: int = 10) -> str:
        """
        Extract clean text content from a webpage using multiple methods
        
        Args:
            url: URL to extract content from
            timeout: Request timeout in seconds
            
        Returns:
            Extracted clean text content or None if failed
        """
        ddg_config = self.config.get('data_sources', {}).get('duckduckgo', {})
        min_length = ddg_config.get('min_content_length', 200)
        extraction_methods = ddg_config.get('extraction_methods', ["trafilatura", "newspaper3k", "beautifulsoup"])
        
        # Try extraction methods in order of preference
        for method in extraction_methods:
            content = None
            
            if method == "trafilatura":
                content = self._extract_with_trafilatura(url, timeout)
            elif method == "newspaper3k":
                content = self._extract_with_newspaper(url, timeout)
            elif method == "readability":
                content = self._extract_with_readability(url, timeout)
            elif method == "beautifulsoup":
                content = self._extract_with_beautifulsoup_enhanced(url, timeout)
            
            if content and len(content.strip()) >= min_length:
                logger.debug(f"{method.title()} extraction successful for {url[:50]} ({len(content)} chars)")
                return content
                
        logger.debug(f"All extraction methods failed for {url[:50]}")
        return None
    
    def _extract_with_trafilatura(self, url: str, timeout: int = 10) -> str:
        """Extract content using trafilatura (best for clean content)"""
        try:
            import trafilatura
            import requests
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, verify=False)
            response.raise_for_status()
            
            # Use trafilatura to extract clean content
            extracted = trafilatura.extract(
                response.content,
                include_comments=False,
                include_tables=True,
                include_formatting=False,
                favor_precision=True,
                url=url
            )
            
            if extracted:
                # Clean and process the text
                cleaned = self._clean_extracted_text(extracted)
                return cleaned
                
            return None
            
        except ImportError:
            logger.debug("trafilatura not available")
            return None
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed for {url}: {e}")
            return None
    
    def _extract_with_newspaper(self, url: str, timeout: int = 10) -> str:
        """Extract content using newspaper3k (good for news articles)"""
        try:
            from newspaper import Article
            
            article = Article(url)
            article.set_config(timeout=timeout, browser_user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            article.download()
            article.parse()
            
            if article.text:
                cleaned = self._clean_extracted_text(article.text)
                return cleaned
                
            return None
            
        except ImportError:
            logger.debug("newspaper3k not available")
            return None
        except Exception as e:
            logger.debug(f"Newspaper3k extraction failed for {url}: {e}")
            return None
    
    def _extract_with_readability(self, url: str, timeout: int = 10) -> str:
        """Extract content using readability-lxml (Mozilla's Readability algorithm)"""
        try:
            import requests
            from readability import Document
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, verify=False)
            response.raise_for_status()
            
            # Use readability to extract main content
            doc = Document(response.content)
            
            # Get the readable content as text
            from bs4 import BeautifulSoup
            readable_html = doc.summary()
            soup = BeautifulSoup(readable_html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            
            if text:
                cleaned = self._clean_extracted_text(text)
                return cleaned
                
            return None
            
        except ImportError:
            logger.debug("readability-lxml not available")
            return None
        except Exception as e:
            logger.debug(f"Readability extraction failed for {url}: {e}")
            return None
    
    def _extract_with_beautifulsoup_enhanced(self, url: str, timeout: int = 10) -> str:
        """Enhanced BeautifulSoup extraction with better content filtering"""
        try:
            import requests
            from bs4 import BeautifulSoup
            import re
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements more aggressively
            unwanted_tags = [
                'script', 'style', 'nav', 'header', 'footer', 'aside', 'form',
                'iframe', 'object', 'embed', 'applet', 'canvas', 'svg',
                'button', 'input', 'select', 'textarea', 'label',
                'noscript', 'meta', 'link', 'base', 'title'
            ]
            
            for tag in soup(unwanted_tags):
                tag.decompose()
            
            # Remove elements with common unwanted classes/ids
            unwanted_selectors = [
                '.advertisement', '.ad', '.ads', '.sidebar', '.menu', '.navigation',
                '.footer', '.header', '.social', '.share', '.comments', '.comment',
                '.related', '.recommended', '.popup', '.modal', '.overlay',
                '#advertisement', '#ad', '#sidebar', '#menu', '#navigation',
                '[class*="ad-"]', '[class*="advertisement"]', '[class*="sidebar"]',
                '[class*="menu"]', '[class*="nav"]', '[id*="ad-"]'
            ]
            
            for selector in unwanted_selectors:
                try:
                    for element in soup.select(selector):
                        element.decompose()
                except:
                    continue
            
            # Prioritized content selectors
            content_selectors = [
                'main article',
                'main .content',
                'article .content',
                'main',
                'article',
                '.post-content',
                '.entry-content',
                '.article-body',
                '.article-content',
                '.content-body',
                '.main-content',
                '.page-content',
                '#main-content',
                '#content',
                '.content'
            ]
            
            main_content = None
            max_text_length = 0
            
            # Find the content section with the most text
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > max_text_length:
                        max_text_length = len(text)
                        main_content = element
            
            # Fallback to body if no main content found
            if not main_content or max_text_length < 200:
                main_content = soup.find('body')
            
            if not main_content:
                return None
            
            # Extract and clean text
            text = main_content.get_text(separator=' ', strip=True)
            cleaned = self._clean_extracted_text(text)
            
            return cleaned if len(cleaned.strip()) > 100 else None
            
        except Exception as e:
            logger.debug(f"Enhanced BeautifulSoup extraction failed for {url}: {e}")
            return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text content"""
        if not text:
            return ""
        
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove repeated characters (like multiple dots, dashes)
        text = re.sub(r'([.]{3,})', '...', text)
        text = re.sub(r'([-]{3,})', '---', text)
        text = re.sub(r'([=]{3,})', '===', text)
        
        # Remove common web garbage patterns
        garbage_patterns = [
            r'cookie\s+policy',
            r'accept\s+cookies',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'subscribe\s+to\s+newsletter',
            r'click\s+here\s+to',
            r'advertisement',
            r'sponsored\s+content',
            r'follow\s+us\s+on',
            r'share\s+this\s+article',
            r'related\s+articles?',
            r'you\s+may\s+also\s+like',
            r'recommended\s+for\s+you',
            r'more\s+from\s+this\s+author',
            r'skip\s+to\s+content',
            r'jump\s+to\s+navigation',
            r'accessibility\s+links?',
        ]
        
        for pattern in garbage_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove single characters on lines (often navigation artifacts)
        text = re.sub(r'\b[a-zA-Z]\b\s*', ' ', text)
        
        # Remove lines with only numbers and symbols
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are mostly non-alphabetic
            if line and len(re.sub(r'[^a-zA-Z]', '', line)) / max(len(line), 1) > 0.3:
                clean_lines.append(line)
        
        text = '\n'.join(clean_lines)
        
        # Final cleanup
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'^\s+|\s+$', '', text)  # Trim whitespace
        
        # Ensure reasonable length limits
        max_length = self.config.get('data_sources', {}).get('duckduckgo', {}).get('max_content_length', 5000)
        if len(text) > max_length:
            # Try to cut at a sentence boundary
            sentences = text[:max_length].split('.')
            if len(sentences) > 1:
                text = '.'.join(sentences[:-1]) + '.'
            else:
                text = text[:max_length]
        
        return text
    
    def _format_duckduckgo_results(self, raw_results: str, query: str) -> List[Dict[str, Any]]:
        """Format DuckDuckGo search results"""
        results = []
        
        try:
            # LangChain DuckDuckGo tool returns a string with results
            # Parse the string format from DuckDuckGoSearchResults
            import json
            import re
            
            # The tool returns results as a string, often in this format:
            # "Title: ... | Link: ... | Snippet: ..."
            # or as JSON-like structure
            
            # Try to parse as JSON first
            try:
                if raw_results.strip().startswith('[') or raw_results.strip().startswith('{'):
                    parsed_results = json.loads(raw_results)
                    if isinstance(parsed_results, list):
                        for item in parsed_results:
                            if isinstance(item, dict):
                                result = {
                                    'source': 'duckduckgo',
                                    'doc_type': 'web_page',
                                    'title': item.get('title', item.get('Title', '')),
                                    'content': item.get('snippet', item.get('Snippet', item.get('content', ''))),
                                    'url': item.get('link', item.get('Link', item.get('url', ''))),
                                    'relevance_score': 0.8
                                }
                                if result['title'] or result['content']:
                                    results.append(result)
                    return results
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Parse text format: look for patterns like "Title: ... Link: ... Snippet: ..."
            entries = re.split(r'\n(?=Title:|Link:|Snippet:)', raw_results)
            
            for entry in entries:
                if not entry.strip():
                    continue
                    
                title_match = re.search(r'Title:\s*(.+?)(?=\s*Link:|$)', entry, re.DOTALL)
                link_match = re.search(r'Link:\s*(.+?)(?=\s*Snippet:|$)', entry, re.DOTALL)
                snippet_match = re.search(r'Snippet:\s*(.+?)$', entry, re.DOTALL)
                
                title = title_match.group(1).strip() if title_match else ''
                url = link_match.group(1).strip() if link_match else ''
                content = snippet_match.group(1).strip() if snippet_match else ''
                
                # Clean up the extracted text
                title = re.sub(r'\s+', ' ', title).strip()
                content = re.sub(r'\s+', ' ', content).strip()
                
                if title or content:
                    result = {
                        'source': 'duckduckgo',
                        'doc_type': 'web_page',
                        'title': title,
                        'content': content,
                        'url': url,
                        'relevance_score': 0.8
                    }
                    results.append(result)
            
            # If still no results, try simpler parsing
            if not results and raw_results.strip():
                # Fallback: treat entire result as content
                result = {
                    'source': 'duckduckgo',
                    'doc_type': 'web_page',
                    'title': f"Search results for: {query}",
                    'content': raw_results[:1000],  # Limit content length
                    'url': '',
                    'relevance_score': 0.5
                }
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error formatting DuckDuckGo results: {e}")
            # Emergency fallback
            if raw_results.strip():
                result = {
                    'source': 'duckduckgo',
                    'doc_type': 'web_page',
                    'title': f"Search results for: {query}",
                    'content': raw_results[:500],
                    'url': '',
                    'relevance_score': 0.3
                }
                results.append(result)
        
        logger.debug(f"Formatted {len(results)} DuckDuckGo results")
        return results
    
    def _format_bing_results(self, raw_results: List[Dict]) -> List[Dict[str, Any]]:
        """Format Bing search results"""
        results = []
        
        for item in raw_results:
            try:
                result = {
                    'source': 'bing',
                    'doc_type': 'web_page',
                    'title': item.get('name', ''),
                    'content': item.get('snippet', ''),
                    'url': item.get('url', ''),
                    'publication_date': item.get('dateLastCrawled', ''),
                    'relevance_score': 0
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to format Bing result: {e}")
                continue
        
        return results
    
    def search_financial_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Search for financial data (placeholder implementation)"""
        # This would integrate with financial data APIs
        financial_data = {}
        
        for symbol in symbols:
            financial_data[symbol] = {
                'symbol': symbol,
                'data': 'Financial data would be fetched from APIs like Alpha Vantage, Yahoo Finance, etc.',
                'timestamp': time.time()
            }
        
        return financial_data
    
    def search_news(self, query: str, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Search news sources (placeholder implementation)"""
        # This would integrate with news APIs
        return [
            {
                'source': 'news_api',
                'doc_type': 'news_article',
                'title': f'News about {query}',
                'content': 'News content would be fetched from news APIs',
                'url': 'https://example.com/news',
                'publication_date': '2024-01-01',
                'relevance_score': 0
            }
        ]
