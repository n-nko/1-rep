import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import hashlib
from datetime import datetime, timedelta
import wikipediaapi
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from functools import wraps

# Configure logging with enhanced format
logging.basicConfig(
    filename="scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
)

@dataclass
class Article:
    title: str
    url: str
    content: Optional[str] = None
    hash: Optional[str] = None
    timestamp: Optional[datetime] = None

def retry_with_backoff(retries=3, backoff_factor=0.3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries:
                        raise e
                    wait_time = backoff_factor * (2 ** i)
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class WebScraper:
    def __init__(self, db_path="articles.db", cache_expiry_days=30):
        self.db_path = db_path
        self.cache_expiry_days = cache_expiry_days
        self.logger = logging.getLogger('WebScraper')
        self.wiki = wikipediaapi.Wikipedia(
            'WebScraper/1.0 (https://example.com; info@example.com)',
            'en'
        )
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.init_db()

    def init_db(self):
        """Initialize database with enhanced schema and indexes"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Articles table with improved schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS articles (
                        id INTEGER PRIMARY KEY,
                        title TEXT NOT NULL,
                        url TEXT UNIQUE NOT NULL,
                        content TEXT,
                        hash TEXT UNIQUE,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        validation_status TEXT,
                        last_updated DATETIME
                    )
                ''')
                
                # Enhanced cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache (
                        url TEXT PRIMARY KEY,
                        content TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        headers TEXT,
                        status_code INTEGER
                    )
                ''')
                
                # Detailed logging table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT,
                        event_description TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        severity TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(hash)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON cache(timestamp)')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def log_event(self, event_type: str, description: str, severity: str = "INFO", metadata: Dict = None):
        """Enhanced logging with metadata support"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO logs (event_type, event_description, severity, metadata) VALUES (?, ?, ?, ?)",
                    (event_type, description, severity, str(metadata) if metadata else None)
                )
                conn.commit()
            
            # Also log to file
            log_message = f"{event_type}: {description}"
            if metadata:
                log_message += f" | Metadata: {metadata}"
            
            if severity == "ERROR":
                self.logger.error(log_message)
            elif severity == "WARNING":
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")

    @retry_with_backoff()
    def clean_cache(self) -> int:
        """Clean expired cache entries with enhanced error handling"""
        try:
            expiry_date = datetime.now() - timedelta(days=self.cache_expiry_days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE timestamp < ?", (expiry_date,))
                deleted_count = cursor.rowcount
                conn.commit()
            
            self.log_event(
                "CACHE_CLEANUP",
                f"Cleaned {deleted_count} expired cache entries",
                metadata={"expired_before": expiry_date.isoformat()}
            )
            return deleted_count
        except sqlite3.Error as e:
            self.log_event(
                "CACHE_CLEANUP_ERROR",
                f"Failed to clean cache: {e}",
                "ERROR"
            )
            raise

    @retry_with_backoff()
    def fetch_articles(self, url: str) -> List[Article]:
        """Fetch articles with improved error handling and rate limiting"""
        articles = []
        try:
            headers = {
                "User-Agent": "WebScraper/1.0 (https://example.com; info@example.com)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Specific selectors for The Verge
            article_selectors = [
                ".duet--content-cards--content-card",
                ".c-compact-river__entry",
                ".c-entry-box--compact",
                "article"
            ]
            
            for selector in article_selectors:
                for article in soup.select(selector):
                    title_tag = article.find(["h1", "h2", "h3"], class_=lambda x: x and 
                        any(word in str(x).lower() for word in ["title", "heading"]))
                    link_tag = article.find("a", href=True)
                    
                    if title_tag and link_tag:
                        title = title_tag.get_text(strip=True)
                        link = link_tag["href"]
                        
                        # Normalize URL
                        if not link.startswith("http"):
                            link = requests.compat.urljoin(url, link)
                        
                        articles.append(Article(title=title, url=link))
                
                if articles:  # If we found articles with current selector, stop trying others
                    break
            
            self.log_event(
                "ARTICLES_FETCHED",
                f"Fetched {len(articles)} articles from {url}",
                metadata={"source_url": url}
            )
            
            return articles
        except requests.exceptions.RequestException as e:
            self.log_event(
                "FETCH_ERROR",
                f"Failed to fetch articles from {url}: {e}",
                "ERROR",
                {"error_type": type(e).__name__}
            )
            raise

    @retry_with_backoff()
    def fetch_content(self, url: str) -> Optional[str]:
        """Fetch content with caching and validation"""
        try:
            # Check cache first
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT content, timestamp FROM cache WHERE url = ?",
                    (url,)
                )
                cached = cursor.fetchone()
                
                if cached:
                    content, timestamp = cached
                    cache_date = datetime.fromisoformat(timestamp)
                    if datetime.now() - cache_date < timedelta(days=self.cache_expiry_days):
                        self.log_event(
                            "CACHE_HIT",
                            f"Using cached content for {url}",
                            metadata={"cache_age": (datetime.now() - cache_date).days}
                        )
                        return content

            # Fetch fresh content
            headers = {
                "User-Agent": "WebScraper/1.0 (https://example.com; info@example.com)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Content selectors for The Verge
            content_selectors = [
                ".duet--article-content-body__content-container",
                ".c-entry-content",
                ".l-col__main",
                "article .content"
            ]
            
            content = ""
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    # Extract text while preserving some structure
                    paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                    content = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
                    if len(content) > 100:  # Minimum content threshold
                        break
            
            if content:
                # Update cache
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (url, content, timestamp, headers, status_code) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)",
                    (url, content, str(headers), response.status_code)
                )
                conn.commit()
                
                self.log_event(
                    "CONTENT_FETCHED",
                    f"Successfully fetched content from {url}",
                    metadata={"content_length": len(content)}
                )
                
                return content
            
            self.log_event(
                "CONTENT_NOT_FOUND",
                f"No suitable content found at {url}",
                "WARNING"
            )
            return None
            
        except Exception as e:
            self.log_event(
                "FETCH_ERROR",
                f"Failed to fetch content from {url}: {e}",
                "ERROR",
                {"error_type": type(e).__name__}
            )
            return None

    def verify_with_wikipedia(self, title: str) -> Tuple[bool, Optional[str]]:
        """Enhanced Wikipedia verification with detailed comparison"""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return False, None
            
            summary = page.summary[:500]
            return True, summary
        except Exception as e:
            self.log_event(
                "WIKI_ERROR",
                f"Failed to verify with Wikipedia: {e}",
                "ERROR",
                {"title": title}
            )
            return False, None

    @retry_with_backoff()
    def save_article(self, article: Article) -> bool:
        """Save article with enhanced validation and error handling"""
        try:
            if not article.content:
                return False
            
            article.hash = hashlib.sha256(article.content.encode()).hexdigest()
            wiki_exists, wiki_summary = self.verify_with_wikipedia(article.title)
            
            validation_status = "verified" if wiki_exists else "unverified"
            if wiki_exists and wiki_summary:
                # Simple content similarity check
                content_start = article.content[:500].lower()
                if not any(phrase.lower() in content_start for phrase in wiki_summary.split()):
                    validation_status = "content_mismatch"
                    self.log_event(
                        "CONTENT_MISMATCH",
                        f"Content mismatch with Wikipedia for {article.title}",
                        "WARNING"
                    )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO articles 
                    (title, url, content, hash, validation_status, last_updated)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (article.title, article.url, article.content,
                     article.hash, validation_status)
                )
                conn.commit()
            
            self.log_event(
                "ARTICLE_SAVED",
                f"Saved article: {article.title}",
                metadata={
                    "validation_status": validation_status,
                    "content_length": len(article.content)
                }
            )
            return True
            
        except sqlite3.IntegrityError as e:
            self.log_event(
                "SAVE_ERROR",
                f"Failed to save article {article.title}: {e}",
                "ERROR"
            )
            return False

    def scrape(self, base_url: str, delay: float = 1.0):
        """Main scraping function with rate limiting"""
        try:
            self.clean_cache()
            articles = self.fetch_articles(base_url)
            
            successful_scrapes = 0
            failed_scrapes = 0
            
            for article in articles:
                try:
                    content = self.fetch_content(article.url)
                    if content:
                        article.content = content
                        if self.save_article(article):
                            successful_scrapes += 1
                        else:
                            failed_scrapes += 1
                    time.sleep(delay)  # Rate limiting
                except Exception as e:
                    failed_scrapes += 1
                    self.log_event(
                        "SCRAPE_ERROR",
                        f"Failed to scrape article {article.title}: {e}",
                        "ERROR"
                    )
            
            self.log_event(
                "SCRAPE_COMPLETED",
                f"Scraping completed for {base_url}",
                metadata={
                    "successful_scrapes": successful_scrapes,
                    "failed_scrapes": failed_scrapes
                }
            )
            
        except Exception as e:
            self.log_event(
                "SCRAPE_ERROR",
                f"Scraping failed for {base_url}: {e}",
                "ERROR"
            )
            raise

if __name__ == "__main__":
    scraper = WebScraper()
    scraper.scrape("https://www.theverge.com/tech")