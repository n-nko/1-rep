import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import hashlib
from datetime import datetime, timedelta
import wikipediaapi
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logging.basicConfig(filename="scraper.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class WebScraper:
    def __init__(self, db_path="articles.db"):
        self.db_path = db_path
        self.init_db()
        self.cache_expiry_days = 30
        self.wiki = wikipediaapi.Wikipedia('en')
        self.categories = {
            "AI": ["machine learning", "neural network", "deep learning", "artificial intelligence", "data science"],
            "Programming": ["Python", "JavaScript", "C++", "coding", "software development", "programming"],
            "Cybersecurity": ["hacking", "malware", "encryption", "security", "cyber attack", "vulnerability"],
            "Technology": ["blockchain", "cloud computing", "IoT", "robotics", "automation"],
            "Science": ["physics", "chemistry", "biology", "research", "scientific"]
        }
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    url TEXT UNIQUE,
                    content TEXT,
                    hash TEXT UNIQUE,
                    category TEXT,
                    similarity_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    url TEXT PRIMARY KEY,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT,
                    severity TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def log_event(self, event, severity="INFO"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO logs (event, severity) VALUES (?, ?)", (event, severity))
            conn.commit()
        logging.info(f"{severity}: {event}")

    def clean_cache(self):
        expiry_date = datetime.now() - timedelta(days=self.cache_expiry_days)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache WHERE timestamp < ?", (expiry_date,))
                deleted_count = cursor.rowcount
                conn.commit()
            self.log_event(f"Cache cleaned: {deleted_count} entries removed")
        except sqlite3.Error as e:
            self.log_event(f"Cache cleaning failed: {str(e)}", "ERROR")

    def fetch_articles(self, url):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.log_event(f"Failed to fetch {url}: {str(e)}", "ERROR")
            return []
        
        soup = BeautifulSoup(response.text, "html.parser")
        articles = []
        
        for article in soup.find_all(["article", "div", "section"]):
            title_tag = article.find(["h1", "h2", "h3"])
            link_tag = article.find("a")
            
            if title_tag and link_tag:
                title = title_tag.get_text(strip=True)
                link = link_tag.get("href")
                if link:
                    if not link.startswith("http"):
                        link = requests.compat.urljoin(url, link)
                    articles.append((title, link))
        
        self.log_event(f"Fetched {len(articles)} articles from {url}")
        return articles

    def fetch_content(self, url):
        self.clean_cache()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM cache WHERE url = ?", (url,))
                cached_content = cursor.fetchone()
                if cached_content:
                    self.log_event(f"Using cached content for {url}")
                    return cached_content[0]
        except sqlite3.Error as e:
            self.log_event(f"Database error while fetching cache: {str(e)}", "ERROR")

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            content_tags = soup.find_all(["p", "article", "section"])
            content = "\n".join(tag.get_text(strip=True) for tag in content_tags if tag.get_text(strip=True))
            
            if len(content) > 100:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT OR REPLACE INTO cache (url, content, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                            (url, content)
                        )
                        conn.commit()
                    self.log_event(f"Cached content for {url}")
                except sqlite3.Error as e:
                    self.log_event(f"Failed to cache content: {str(e)}", "ERROR")
                return content
            
            return None
        except requests.exceptions.RequestException as e:
            self.log_event(f"Failed to fetch content from {url}: {str(e)}", "ERROR")
            return None

    def verify_with_wikipedia(self, title, content):
        try:
            page = self.wiki.page(title)
            if page.exists():
                wiki_summary = page.summary[:500]
                if not content or not wiki_summary:
                    return None
                
                # Calculate TF-IDF similarity
                try:
                    tfidf_matrix = self.vectorizer.fit_transform([content[:500], wiki_summary])
                    similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]
                    
                    if similarity < 0.3:
                        self.log_event(f"Low similarity ({similarity:.2f}) between article and Wikipedia: {title}", "WARNING")
                    return {
                        'summary': wiki_summary,
                        'similarity': similarity
                    }
                except Exception as e:
                    self.log_event(f"Error calculating similarity: {str(e)}", "ERROR")
                    return None
        except Exception as e:
            self.log_event(f"Wikipedia verification failed for {title}: {str(e)}", "ERROR")
        return None

    def categorize_article(self, content):
        if not content:
            return "Uncategorized"
        
        category_scores = {}
        content_lower = content.lower()
        
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                pattern = rf"\b{re.escape(keyword.lower())}\b"
                matches = len(re.findall(pattern, content_lower))
                score += matches
            category_scores[category] = score
        
        if not any(category_scores.values()):
            return "Uncategorized"
        
        return max(category_scores.items(), key=lambda x: x[1])[0]

    def save_article(self, title, url, content):
        if not all([title, url, content]):
            self.log_event(f"Invalid article data: title={bool(title)}, url={bool(url)}, content={bool(content)}", "ERROR")
            return False

        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            wiki_verification = self.verify_with_wikipedia(title, content)
            category = self.categorize_article(content)
            similarity_score = wiki_verification['similarity'] if wiki_verification else 0.0

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO articles 
                    (title, url, content, hash, category, similarity_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (title, url, content, content_hash, category, similarity_score))
                conn.commit()
                
            self.log_event(
                f"Saved article: {title} | Category: {category} | "
                f"Similarity Score: {similarity_score:.2f}"
            )
            return True
            
        except sqlite3.IntegrityError:
            self.log_event(f"Duplicate article detected: {url}", "WARNING")
            return False
        except Exception as e:
            self.log_event(f"Failed to save article {title}: {str(e)}", "ERROR")
            return False