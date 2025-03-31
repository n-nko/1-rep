import os
import time
import json
import sqlite3
import logging
import numpy as np
from typing import Optional, Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('JARVIS')

class JARVIS:
    def __init__(self):
        """Initialize JARVIS with enhanced learning capabilities"""
        self.db_path = "knowledge.db"
        self.model_path = "trained_model.h5"
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.knowledge_cache = {}
        self.initialize_database()
        self.load_or_create_model()

    def initialize_database(self):
        """Initialize the knowledge database with improved schema and connection pooling"""
        try:
            # Use a timeout to handle concurrent access
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
                
                cursor = conn.cursor()
                
                # Create tables with better structure and indexes
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY,
                    category TEXT NOT NULL,
                    source TEXT,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    last_validated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create indexes for frequently queried columns
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge(confidence)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_last_validated ON knowledge(last_validated)")
                
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys = ON")
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id INTEGER PRIMARY KEY,
                    description TEXT,
                    code TEXT,
                    language TEXT,
                    efficiency_score REAL,
                    last_tested TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_history (
                    id INTEGER PRIMARY KEY,
                    knowledge_id INTEGER,
                    validation_score REAL,
                    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(knowledge_id) REFERENCES knowledge(id)
                )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def load_or_create_model(self):
        """Load existing model or create a new one with enhanced architecture"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info("Loaded existing model")
            else:
                self.create_new_model()
                logger.info("Created new model")
                if self._get_training_data()[0]:
                    self.train_models()
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

    def _get_training_data(self):
        """Retrieve validated knowledge entries for training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT content, confidence 
                    FROM knowledge 
                    WHERE confidence > 0.7
                      AND datetime(last_validated) > datetime('now', '-7 days')
                """)
                data = cursor.fetchall()
                return [x[0] for x in data], [x[1] for x in data]
        except sqlite3.Error as e:
            logger.error(f"Training data error: {e}")
            return [], []

    def train_models(self):
        """Train both TF-IDF vectorizer and neural network model"""
        contents, confidences = self._get_training_data()
        if not contents:
            logger.warning("No training data available")
            return

        # Train and save vectorizer
        self.vectorizer.fit(contents)
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')
        
        # Convert text to features
        X = self.vectorizer.transform(contents).toarray()
        y = np.array(confidences)

        # Train model with early stopping
        self.model.fit(X, y, 
                      epochs=50, 
                      batch_size=32,
                      validation_split=0.2,
                      callbacks=[keras.callbacks.EarlyStopping(patience=3)])
        
        # Save updated model
        self.model.save(self.model_path)
        logger.info("Model training completed with %d samples", len(contents))

    def create_new_model(self):
        """Create a new neural network model with improved architecture"""
        self.model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(1000,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def validate_knowledge(self, content: str) -> float:
        """Validate knowledge using the trained model"""
        try:
            # Convert content to feature vector
            features = self.vectorizer.transform([content]).toarray()
            
            # Get model prediction
            confidence = float(self.model.predict(features)[0][0])
            
            return confidence
        except Exception as e:
            logger.error(f"Knowledge validation error: {e}")
            return 0.0

    def save_knowledge(self, category: str, source: str, content: str) -> bool:
        """Save new knowledge with validation"""
        try:
            # Validate the knowledge
            confidence = self.validate_knowledge(content)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO knowledge (category, source, content, confidence, last_validated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (category, source, content, confidence))
                
                knowledge_id = cursor.lastrowid
                
                # Record validation history
                cursor.execute("""
                INSERT INTO validation_history (knowledge_id, validation_score)
                VALUES (?, ?)
                """, (knowledge_id, confidence))
                
                conn.commit()
                
                # Update cache
                self.knowledge_cache[category] = self.knowledge_cache.get(category, []) + [(content, confidence)]
                
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving knowledge: {e}")
            return False

    def process_message(self, message: str, context_id: str) -> Optional[str]:
        """Process user message with context awareness and enhanced response generation"""
        try:
            # Analyze message context and category
            category = self._categorize_message(message)
            
            # Get relevant knowledge
            knowledge = self._get_relevant_knowledge(category, message)
            
            # Generate response using the knowledge
            response = self._generate_response(message, knowledge, context_id)
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None

    def _categorize_message(self, message: str) -> str:
        """Categorize incoming message using TF-IDF and clustering"""
        # Simple categorization based on keywords for now
        # This can be enhanced with more sophisticated ML techniques
        keywords = {
            'technical': ['code', 'programming', 'debug', 'error', 'function'],
            'conceptual': ['explain', 'what', 'how', 'why', 'concept'],
            'task': ['create', 'build', 'implement', 'solve', 'make']
        }
        
        message_lower = message.lower()
        max_category = 'general'
        max_count = 0
        
        for category, words in keywords.items():
            count = sum(1 for word in words if word in message_lower)
            if count > max_count:
                max_count = count
                max_category = category
        
        return max_category

    def _get_relevant_knowledge(self, category: str, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from database with enhanced caching and query optimization"""
        try:
            cache_key = f"{category}_{hash(query)}"
            
            # Check cache first with TTL of 5 minutes
            if cache_key in self.knowledge_cache:
                cache_time, cache_data = self.knowledge_cache[cache_key]
                if time.time() - cache_time < 300:  # 5 minutes TTL
                    return cache_data
            
            # Use connection with timeout and optimized query
            with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                conn.execute("PRAGMA temp_store = MEMORY")
                conn.execute("PRAGMA cache_size = 10000")
                
                cursor = conn.cursor()
                cursor.execute("""
                SELECT k.content, k.confidence, MAX(v.validation_score) as latest_validation
                FROM knowledge k
                LEFT JOIN validation_history v ON k.id = v.knowledge_id
                WHERE k.category = ? 
                  AND k.confidence > 0.7
                  AND (k.last_validated IS NULL OR 
                       datetime(k.last_validated) > datetime('now', '-7 days'))
                GROUP BY k.id
                ORDER BY k.confidence DESC, latest_validation DESC
                LIMIT 5
                """, (category,))
                
                results = [(row[0], row[1]) for row in cursor.fetchall()]
                
                # Update cache with timestamp
                self.knowledge_cache[cache_key] = (time.time(), results)
                
                # Cleanup old cache entries
                self._cleanup_cache()
                
                return results
        except sqlite3.Error as e:
            logger.error(f"Error retrieving knowledge: {e}")
            return []
            
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [k for k, v in self.knowledge_cache.items()
                       if current_time - v[0] > 300]
        for k in expired_keys:
            del self.knowledge_cache[k]

    def _generate_response(self, message: str, knowledge: List[Dict[str, Any]], context_id: str) -> str:
        """Generate response using retrieved knowledge and context"""
        try:
            # Simple response generation for now
            # This can be enhanced with more sophisticated NLP techniques
            if not knowledge:
                return "I don't have enough information to provide a good answer. Could you please provide more context?"
            
            # Use the most confident knowledge as base response
            best_knowledge = max(knowledge, key=lambda x: x[1])
            response = best_knowledge[0]
            
            # Add context awareness
            response = f"Based on my knowledge, {response}"
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."