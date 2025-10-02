# src/web_search_engine.py
import random
from urllib.parse import quote

class WebSearchEngine:
    """Handles web search operations"""
    def __init__(self):
        self.search_engines = {
            'mock': self._search_mock,
            'google': self._search_google,
            'bing': self._search_bing
        }
        self.current_engine = 'mock'  # Default to mock for testing
    
    def search(self, query, num_results=10):
        """Search the web using the configured engine"""
        if self.current_engine in self.search_engines:
            return self.search_engines[self.current_engine](query, num_results)
        return []
    
    def _search_mock(self, query, num_results):
        """Mock search engine for testing purposes"""
        # Generate mock results based on query keywords
        keywords = query.lower().split()
        mock_results = []
        
        # Common web sources
        web_sources = [
            {
                'title': 'Wikipedia - General Knowledge',
                'url': 'https://en.wikipedia.org/wiki/Knowledge',
                'content': 'Knowledge is a familiarity, awareness, or understanding of someone or something.'
            },
            {
                'title': 'Famous Quotes Database',
                'url': 'https://example.com/quotes',
                'content': 'A collection of famous quotes from around the world.'
            },
            {
                'title': 'Common Phrases Dictionary',
                'url': 'https://example.com/phrases',
                'content': 'A comprehensive dictionary of common English phrases and their origins.'
            },
            {
                'title': 'Academic Papers Repository',
                'url': 'https://example.com/papers',
                'content': 'Access to millions of academic papers and research articles.'
            },
            {
                'title': 'News Articles Archive',
                'url': 'https://example.com/news',
                'content': 'Archive of news articles from various sources around the world.'
            }
        ]
        
        # Generate results based on query
        for i in range(min(num_results, len(web_sources))):
            source = web_sources[i]
            result = {
                'title': source['title'],
                'url': source['url'],
                'content': source['content'] + ' ' + ' '.join(keywords[:3])
            }
            mock_results.append(result)
        
        return mock_results
    
    def _search_google(self, query, num_results):
        """Search using Google (requires API key)"""
        # Placeholder for real Google search implementation
        return self._search_mock(query, num_results)
    
    def _search_bing(self, query, num_results):
        """Search using Bing (requires API key)"""
        # Placeholder for real Bing search implementation
        return self._search_mock(query, num_results)