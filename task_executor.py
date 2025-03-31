import logging
from typing import Dict, Any, List, Optional
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.feature_extraction.text import TfidfVectorizer
from task_processor import TaskProcessor
from knowledge_manager import KnowledgeManager
from task_optimizer import TaskOptimizer

class TaskExecutor:
    def __init__(self, knowledge_db_path: str = 'knowledge.db'):
        """Initialize TaskExecutor with ML capabilities and knowledge integration"""
        self.task_processor = TaskProcessor()
        self.knowledge_manager = KnowledgeManager(knowledge_db_path)
        self.task_optimizer = TaskOptimizer(knowledge_db_path)
        self.execution_history = []

    def execute_task(self, task_description: str, solution: Optional[Any] = None) -> Dict[str, Any]:
        """Execute a task with knowledge-based optimization and validation"""
        try:
            # Process task and get execution plan
            task_result = self.task_processor.process_task(task_description)
            
            # Find related knowledge
            related_knowledge = self.knowledge_manager.find_related_knowledge(
                task_description, threshold=0.6
            )
            
            # If solution provided, optimize it
            if solution:
                optimization_result = self.task_optimizer.optimize_task_solution(
                    task_description, solution
                )
                execution_result = self._execute_optimized_solution(
                    task_description,
                    optimization_result,
                    related_knowledge
                )
            else:
                # Generate and execute solution based on task and knowledge
                execution_result = self._generate_and_execute_solution(
                    task_description,
                    task_result,
                    related_knowledge
                )
            
            self.execution_history.append(execution_result)
            return execution_result
            
        except Exception as e:
            logging.error(f"Error in task execution: {e}")
            return {
                'task_description': task_description,
                'status': 'error',
                'error_message': str(e)
            }

    def _execute_optimized_solution(self, task_description: str,
                                   optimization_result: Dict[str, Any],
                                   related_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute an optimized solution with knowledge integration"""
        try:
            # Validate optimized solution
            validation_result = self.task_processor.validate_solution(
                str(hash(task_description)),  # Generate unique task ID
                optimization_result['original_solution']
            )
            
            execution_result = {
                'task_description': task_description,
                'execution_status': 'completed' if validation_result['is_valid'] else 'failed',
                'optimization_score': optimization_result['optimization_score'],
                'validation_score': validation_result['validation_score'],
                'performance_metrics': optimization_result['performance_metrics'],
                'knowledge_applications': [
                    {
                        'source': item['source'],
                        'similarity_score': item['similarity_score'],
                        'applied_improvement': item['suggested_improvement']
                    } for item in optimization_result.get('knowledge_based_improvements', [])
                ],
                'feedback': validation_result['feedback']
            }
            
            # Store execution result as knowledge if successful
            if validation_result['is_valid']:
                self.knowledge_manager.add_knowledge(
                    content=str(optimization_result['original_solution']),
                    source=f'task_execution_{hash(task_description)}',
                    category='successful_executions'
                )
            
            return execution_result
            
        except Exception as e:
            logging.error(f"Error executing optimized solution: {e}")
            return {
                'task_description': task_description,
                'status': 'error',
                'error_message': str(e)
            }

    def _generate_and_execute_solution(self, task_description: str,
                                      task_result: Dict[str, Any],
                                      related_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate and execute solution based on task analysis and knowledge"""
        try:
            # Generate initial solution based on task complexity
            if task_result['complexity_score'] < 0.5:
                # Simple task, use direct execution
                solution = self._generate_simple_solution(task_description, related_knowledge)
            else:
                # Complex task, use knowledge-based solution generation
                solution = self._generate_complex_solution(task_description, task_result, related_knowledge)
            
            # Optimize and execute the generated solution
            return self._execute_optimized_solution(
                task_description,
                self.task_optimizer.optimize_task_solution(task_description, solution),
                related_knowledge
            )
            
        except Exception as e:
            logging.error(f"Error generating and executing solution: {e}")
            return {
                'task_description': task_description,
                'status': 'error',
                'error_message': str(e)
            }

    def _generate_simple_solution(self, task_description: str,
                                 related_knowledge: List[Dict[str, Any]]) -> Any:
        """Generate solution for simple tasks"""
        # Use most similar knowledge item as solution template
        if related_knowledge:
            return related_knowledge[0]['content']
        return task_description  # Fallback to basic execution

    def _generate_complex_solution(self, task_description: str,
                                 task_result: Dict[str, Any],
                                 related_knowledge: List[Dict[str, Any]]) -> Any:
        """Generate solution for complex tasks using knowledge integration"""
        solution_components = []
        
        # Integrate knowledge from related items
        for knowledge_item in related_knowledge[:3]:  # Use top 3 related items
            if knowledge_item['similarity_score'] > 0.7:
                solution_components.append(knowledge_item['content'])
        
        # Combine components based on task execution plan
        return '\n'.join(solution_components) if solution_components else task_description

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the history of task executions"""
        return self.execution_history

    def analyze_execution_performance(self) -> Dict[str, Any]:
        """Analyze performance trends in execution history"""
        if not self.execution_history:
            return {'status': 'no_data', 'message': 'No execution history available'}
        
        try:
            successful_executions = [
                item for item in self.execution_history
                if item.get('execution_status') == 'completed'
            ]
            
            return {
                'total_executions': len(self.execution_history),
                'successful_executions': len(successful_executions),
                'success_rate': len(successful_executions) / len(self.execution_history) \
                    if self.execution_history else 0,
                'average_optimization_score': np.mean([
                    item.get('optimization_score', 0) for item in self.execution_history
                ]),
                'status': 'success'
            }
        except Exception as e:
            logging.error(f"Error analyzing execution performance: {e}")
            return {'status': 'error', 'error_message': str(e)}