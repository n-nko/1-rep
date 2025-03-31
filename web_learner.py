import logging
from enhanced_scraper import WebScraper
from assistant_enhanced import JARVIS
from typing import Optional, Dict, Any

class WebLearner:
    def __init__(self):
        self.scraper = WebScraper()
        self.jarvis = JARVIS()
        self.logger = logging.getLogger('WebLearner')
    
    def learn_from_website(self, url: str) -> Dict[str, Any]:
        """Learn from a given website by scraping its content and processing it with JARVIS"""
        try:
            # First, scrape articles from the website
            articles = self.scraper.fetch_articles(url)
            if not articles:
                return {
                    'success': False,
                    'error': 'No articles found on the website',
                    'learned_facts': []
                }
            
            learned_facts = []
            for article in articles:
                # Fetch full content for each article
                content = self.scraper.fetch_content(article.url)
                if not content:
                    continue
                
                # Process content with JARVIS to extract knowledge
                article_facts = self.process_content(article.title, content)
                if article_facts:
                    learned_facts.extend(article_facts)
                
                # Save article for future reference
                article.content = content
                self.scraper.save_article(article)
            
            return {
                'success': True,
                'error': None,
                'learned_facts': learned_facts
            }
            
        except Exception as e:
            self.logger.error(f'Failed to learn from website {url}: {e}')
            return {
                'success': False,
                'error': str(e),
                'learned_facts': []
            }
    
    def process_content(self, title: str, content: str) -> Optional[list]:
        """Process article content to extract facts and knowledge"""
        try:
            # First verify content with Wikipedia if possible
            is_verified, wiki_summary = self.scraper.verify_with_wikipedia(title)
            
            # Prepare content for JARVIS processing
            context = f"Title: {title}\n\nContent: {content}"
            if is_verified and wiki_summary:
                context += f"\n\nWikipedia Summary: {wiki_summary}"
            
            # Process with JARVIS and extract facts
            response = self.jarvis.process_message(
                f"Please analyze this article and extract key facts and knowledge:\n\n{context}"
            )
            
            # Return extracted facts
            return [fact.strip() for fact in response.split('\n') if fact.strip()]
            
        except Exception as e:
            self.logger.error(f'Failed to process content for {title}: {e}')
            return None