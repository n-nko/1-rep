import chromadb
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from threading import Lock

class ChromaMemory:
    def __init__(self, path: str = "./memory_db", max_retries: int = 3):
        """Initialize ChromaDB client with persistent storage and enhanced error handling"""
        self.path = path
        self.max_retries = max_retries
        self.client = None
        self.collection = None
        self.lock = Lock()
        self.logger = logging.getLogger('ChromaMemory')
        
        # Initialize with retry mechanism
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                self.client = chromadb.PersistentClient(path=path)
                self.collection = self.client.get_or_create_collection(
                    name="knowledge",
                    metadata={
                        "description": "Long-term knowledge storage with vector embeddings",
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                self.logger.info("ChromaDB initialized successfully")
                return
            except Exception as e:
                last_exception = e
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                self.logger.warning(f"Initialization attempt {retry_count} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        self.logger.error(f"Failed to initialize ChromaDB after {self.max_retries} attempts")
        raise last_exception
    
    @contextmanager
    def _db_operation(self):
        """Context manager for database operations with automatic retry and error handling"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                with self.lock:
                    yield
                return
            except Exception as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    self.logger.error(f"Operation failed after {self.max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** retry_count
                self.logger.warning(f"Operation attempt {retry_count} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def add_fact(self, fact: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a fact to the knowledge base with metadata and enhanced error handling"""
        try:
            # Generate a unique ID for the fact
            fact_id = str(hash(fact))
            
            # Add metadata if not provided
            if metadata is None:
                metadata = {
                    "type": "fact",
                    "source": "user_interaction",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0"
                }
            
            # Add the document to the collection with retry mechanism
            with self._db_operation():
                self.collection.add(
                    documents=[fact],
                    metadatas=[metadata],
                    ids=[fact_id]
                )
                self.logger.info(f"Added fact with ID: {fact_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to add fact: {e}")
            return False
    
    def search_facts(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant facts using semantic similarity"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents']:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    formatted_results.append({
                        'fact': doc,
                        'metadata': metadata,
                        'relevance_score': 1 - (distance / 2)  # Normalize distance to score
                    })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Failed to search facts: {e}")
            return []
    
    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Retrieve all facts from the knowledge base"""
        try:
            results = self.collection.get()
            formatted_results = []
            
            if results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    formatted_results.append({
                        'fact': doc,
                        'metadata': metadata
                    })
            
            return formatted_results
        except Exception as e:
            self.logger.error(f"Failed to retrieve facts: {e}")
            return []
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact from the knowledge base"""
        try:
            self.collection.delete(ids=[fact_id])
            self.logger.info(f"Deleted fact with ID: {fact_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete fact: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all facts from the knowledge base"""
        try:
            self.client.delete_collection("knowledge")
            self.collection = self.client.create_collection(
                name="knowledge",
                metadata={"description": "Long-term knowledge storage with vector embeddings"}
            )
            self.logger.info("Cleared all facts from knowledge base")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear knowledge base: {e}")
            return False