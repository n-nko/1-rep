import numpy as np
import tensorflow as tf
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer, AutoModel

class CodeGeneratorUtils:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_pipeline = pipeline('feature-extraction', model='bert-base-uncased')
        
    def _vectorize_context(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context dictionary into a vector representation using BERT"""
        # Prepare context string
        context_str = f"Category: {context['category']}\n"
        context_str += f"Dependencies: {', '.join(context['dependencies'])}\n"
        context_str += f"Previous code:\n{context['previous_code']}"
        
        # Get BERT embeddings
        embeddings = self.bert_pipeline(context_str)
        
        # Average pooling of token embeddings
        context_vector = np.mean(embeddings[0], axis=0)
        return context_vector
    
    def _optimize_prompt(self, description: str, context_vector: np.ndarray) -> str:
        """Optimize the prompt using context vector and task description"""
        # Encode the description
        desc_embedding = self.bert_pipeline(description)[0]
        desc_vector = np.mean(desc_embedding, axis=0)
        
        # Combine vectors using attention mechanism
        attention_weights = tf.nn.softmax(
            tf.matmul(tf.expand_dims(desc_vector, 0), 
                     tf.expand_dims(context_vector, 1))
        )
        
        # Create optimized prompt
        prompt = f"Task description: {description}\n"
        prompt += "Generate code that:\n"
        prompt += "1. Follows best practices and design patterns\n"
        prompt += "2. Includes proper error handling\n"
        prompt += "3. Is well-documented\n"
        prompt += "4. Is efficient and maintainable\n"
        
        return prompt
    
    def _post_process_code(self, code: str, category: str) -> str:
        """Post-process and optimize generated code"""
        # Remove unnecessary comments and empty lines
        lines = [line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        
        # Add appropriate imports based on category
        imports = []
        if category == 'database':
            imports.extend([
                'import sqlite3',
                'from typing import Dict, Any, List',
                'import logging'
            ])
        elif category == 'api':
            imports.extend([
                'from fastapi import FastAPI, HTTPException',
                'from pydantic import BaseModel',
                'import uvicorn'
            ])
        elif category == 'interface':
            imports.extend([
                'import tkinter as tk',
                'from tkinter import ttk',
                'import asyncio'
            ])
        
        # Add error handling if not present
        if 'try:' not in code:
            processed_code = 'try:\n    ' + '\n    '.join(lines)
            processed_code += '\nexcept Exception as e:\n    logging.error(f"Error: {e}")';
        else:
            processed_code = '\n'.join(lines)
        
        # Combine imports and processed code
        final_code = '\n'.join(imports) + '\n\n' + processed_code
        
        return final_code