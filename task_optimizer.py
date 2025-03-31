import logging
from typing import Dict, Any, List, Optional
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.feature_extraction.text import TfidfVectorizer
from task_processor import TaskProcessor
from knowledge_manager import KnowledgeManager

class TaskOptimizer:
    def __init__(self, knowledge_db_path: str = 'knowledge.db'):
        """Initialize TaskOptimizer with ML capabilities and knowledge integration"""
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = self._create_optimization_model()
        self.task_processor = TaskProcessor()
        self.knowledge_manager = KnowledgeManager(knowledge_db_path)
        self.optimization_history = []

    def _create_optimization_model(self) -> keras.Model:
        """Create an enhanced neural network model for task optimization"""
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(1000,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        return model

    def optimize_task_solution(self, task_description: str, solution: Any) -> Dict[str, Any]:
        """Optimize task solution using ML and knowledge base"""
        try:
            # Get task complexity and initial optimization from task processor
            task_result = self.task_processor.process_task(task_description)
            initial_optimization = self.task_processor.optimize_solution(solution)
            
            # Find related knowledge
            related_knowledge = self.knowledge_manager.find_related_knowledge(
                str(solution), threshold=0.7
            )
            
            # Vectorize solution for model processing
            solution_vector = self.vectorizer.transform([str(solution)]).toarray()
            optimization_score = float(self.model.predict(solution_vector)[0][0])
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(
                solution,
                optimization_score,
                related_knowledge
            )
            
            # Combine all optimization results
            optimization_result = {
                'task_description': task_description,
                'original_solution': solution,
                'optimization_score': optimization_score,
                'suggestions': optimization_suggestions,
                'performance_metrics': initial_optimization.get('improvement_metrics', {}),
                'knowledge_based_improvements': [
                    {
                        'source': f"knowledge_item_{item['id']}",
                        'similarity_score': item['similarity_score'],
                        'suggested_improvement': item['content'][:200]
                    } for item in related_knowledge[:3]
                ] if related_knowledge else []
            }
            
            self.optimization_history.append(optimization_result)
            return optimization_result
            
        except Exception as e:
            logging.error(f"Error in task optimization: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'task_description': task_description
            }

    def _generate_optimization_suggestions(self, solution: Any, 
                                         optimization_score: float,
                                         related_knowledge: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization suggestions based on solution analysis"""
        suggestions = []
        
        # Basic optimization suggestions
        if optimization_score < 0.5:
            suggestions.extend([
                "Consider restructuring the solution for better efficiency",
                "Review algorithm complexity and optimize if possible",
                "Look for potential bottlenecks in the implementation"
            ])
        elif optimization_score < 0.8:
            suggestions.extend([
                "Minor optimizations possible",
                "Consider edge cases for improved robustness",
                "Review error handling implementation"
            ])
        
        # Add knowledge-based suggestions
        if related_knowledge:
            for item in related_knowledge[:2]:  # Use top 2 related items
                if item['similarity_score'] > 0.8:
                    suggestions.append(
                        f"Consider similar solution pattern from knowledge base "
                        f"(similarity: {item['similarity_score']:.2f})"
                    )
        
        return suggestions

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of optimization attempts"""
        return self.optimization_history

    def analyze_optimization_trends(self) -> Dict[str, Any]:
        """Analyze trends in optimization history"""
        if not self.optimization_history:
            return {'status': 'no_data', 'message': 'No optimization history available'}
        
        try:
            scores = [item['optimization_score'] for item in self.optimization_history 
                     if 'optimization_score' in item]
            
            return {
                'average_score': np.mean(scores) if scores else 0,
                'score_trend': 'improving' if len(scores) > 1 and 
                               scores[-1] > np.mean(scores[:-1]) else 'stable',
                'total_optimizations': len(self.optimization_history),
                'status': 'success'
            }
        except Exception as e:
            logging.error(f"Error analyzing optimization trends: {e}")
            return {'status': 'error', 'error_message': str(e)}