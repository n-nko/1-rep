import logging
from typing import Dict, Any, List
from task_planner import TaskPlanner, SubTask
from code_generator import CodeGenerator, ModelType
from code_validator import CodeValidator

class TaskIntegrator:
    def __init__(self, code_llama_path: str, starcoder_path: str = None):
        self.logger = logging.getLogger('TaskIntegrator')
        
        # Initialize components
        self.task_planner = TaskPlanner(self.logger)
        self.code_generator = CodeGenerator(code_llama_path, starcoder_path)
        self.code_validator = CodeValidator()
        
        # Task execution state
        self.current_task = None
        self.generated_components = {}
        self.integration_status = {}
        
    def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a complete task with subtask generation and integration"""
        try:
            self.logger.info(f'Starting task execution: {task_description}')
            self.current_task = task_description
            
            # Decompose task into subtasks
            subtasks = self.task_planner.decompose_task(task_description)
            if not subtasks:
                return {
                    'success': False,
                    'error': 'Failed to decompose task into subtasks'
                }
                
            # Process each subtask
            for subtask in subtasks:
                result = self._process_subtask(subtask)
                if not result['success']:
                    return result
                    
            # Integrate all components
            integration_result = self._integrate_components(subtasks)
            
            return integration_result
            
        except Exception as e:
            error_context = {
                'task_description': task_description,
                'current_subtasks': [st.id for st in (subtasks or [])],
                'generated_components': list(self.generated_components.keys())
            }
            self.logger.error(f'Task execution failed: {e}', extra={'error_context': error_context})
            return self._create_error_response(f'Task execution failed: {str(e)}. Context: {error_context}')
            
    def _process_subtask(self, subtask: SubTask) -> Dict[str, Any]:
        """Process a single subtask with code generation and validation"""
        try:
            self.logger.info(f'Processing subtask: {subtask.id} - {subtask.description}')
            
            # Check dependencies
            for dep_id in subtask.dependencies:
                if dep_id not in self.generated_components:
                    return {
                        'success': False,
                        'error': f'Dependency {dep_id} not satisfied'
                    }
                    
            # Select appropriate model based on task category
            model_type = ModelType.CODE_LLAMA
            if subtask.category in ['api', 'interface']:
                model_type = ModelType.STARCODER
                
            # Generate code
            generation_result = self.code_generator.generate_code(
                task_description=subtask.description,
                category=subtask.category,
                model_type=model_type,
                existing_code=self._get_dependency_code(subtask)
            )
            
            if not generation_result['success']:
                return generation_result
                
            # Validate generated code
            validation_result = self.code_validator.validate_code(
                code=generation_result['code'],
                category=subtask.category
            )
            
            if not validation_result['success']:
                # Try to improve code based on validation results
                improved_code = self.code_generator.improve_code(
                    code=generation_result['code'],
                    validation_result=validation_result
                )
                
                # Validate improved code
                validation_result = self.code_validator.validate_code(
                    code=improved_code,
                    category=subtask.category
                )
                
                if not validation_result['success']:
                    return {
                        'success': False,
                        'error': f'Code validation failed: {validation_result["errors"]}'
                    }
                    
                generation_result['code'] = improved_code
                
            # Store generated component
            self.generated_components[subtask.id] = {
                'code': generation_result['code'],
                'category': subtask.category,
                'validation': validation_result
            }
            
            return {'success': True}
            
        except Exception as e:
            self.logger.error(f'Subtask processing failed: {e}')
            return self._create_error_response(str(e))
            
    def _get_dependency_code(self, subtask: SubTask) -> str:
        """Get code from dependencies for context"""
        dependency_code = ''
        
        for dep_id in subtask.dependencies:
            if dep_id in self.generated_components:
                dependency_code += f"\n# Code from {dep_id}\n"
                dependency_code += self.generated_components[dep_id]['code']
                
        return dependency_code

    def _validate_dependencies(self, subtask: SubTask) -> Dict[str, Any]:
        """Validate subtask dependencies with detailed checking"""
        missing_deps = []
        for dep_id in subtask.dependencies:
            if dep_id not in self.generated_components:
                missing_deps.append(dep_id)
                
        if missing_deps:
            return self._create_error_response(
                f'Dependencies not satisfied: {", ".join(missing_deps)}'
            )
        return {'success': True}
    
    def _select_model_type(self, subtask: SubTask, related_knowledge: List[Dict[str, Any]]) -> ModelType:
        """Select appropriate model type based on task category and context"""
        if subtask.category in ['api', 'interface']:
            return ModelType.STARCODER
        if any(k.get('category') == 'ml_model' for k in related_knowledge):
            return ModelType.CODE_LLAMA
        return ModelType.CODE_LLAMA
    
    def _validate_generated_code(self, code: str, category: str) -> Dict[str, Any]:
        """Validate generated code with enhanced checking"""
        return self.code_validator.validate_code(
            code=code,
            category=category,
            strict_mode=True
        )
    
    def _improve_and_validate_code(self, code: str, validation_result: Dict[str, Any], subtask: SubTask) -> Dict[str, Any]:
        """Improve and validate code with multiple attempts"""
        max_attempts = 3
        current_attempt = 0
        
        while current_attempt < max_attempts:
            improved_code = self.code_generator.improve_code(
                code=code,
                validation_result=validation_result,
                context={
                    'subtask': subtask,
                    'attempt': current_attempt + 1
                }
            )
            
            new_validation = self._validate_generated_code(
                improved_code,
                subtask.category
            )
            
            if new_validation['success']:
                return {
                    'success': True,
                    'code': improved_code,
                    'validation': new_validation
                }
                
            current_attempt += 1
            validation_result = new_validation
            
        return self._create_error_response(
            f'Failed to improve code after {max_attempts} attempts: '
            f'{validation_result.get("errors", ["Unknown error"])}'
        )
    
    def _handle_generation_failure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation failure with detailed error reporting"""
        error_msg = result.get('error', 'Unknown generation error')
        self.logger.error(f'Code generation failed: {error_msg}')
        return self._create_error_response(f'Code generation failed: {error_msg}')
    
    def _handle_subtask_failure(self, subtask: SubTask, result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subtask processing failure with context"""
        error_msg = result.get('error', 'Unknown error')
        self.logger.error(f'Subtask {subtask.id} failed: {error_msg}')
        return {
            'success': False,
            'error': f'Subtask {subtask.id} failed: {error_msg}',
            'subtask': subtask.id,
            'context': result.get('context', {})
        }
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_msg
        }
    
    def _store_execution_metrics(self, result: Dict[str, Any]) -> None:
        """Store execution metrics for analysis"""
        self.execution_metrics.append({
            'task_id': hash(self.current_task),
            'success': result['success'],
            'components_count': len(self.generated_components),
            'validation_scores': [
                comp['validation'].get('score', 0)
                for comp in self.generated_components.values()
            ],
            'performance_metrics': result.get('performance_metrics', {})
        })
    
    def get_execution_metrics(self) -> List[Dict[str, Any]]:
        """Get execution metrics for analysis"""
        return self.execution_metrics
    
    def reset_state(self) -> None:
        """Reset integrator state for new task"""
        self.current_task = None
        self.generated_components = {}
        self.integration_status = {}
        
    def _integrate_components(self, subtasks: List[SubTask]) -> Dict[str, Any]:
        """Integrate all generated components into final solution"""
        try:
            self.logger.info('Starting component integration')
            
            # Validate all components are generated
            for subtask in subtasks:
                if subtask.id not in self.generated_components:
                    return {
                        'success': False,
                        'error': f'Missing component: {subtask.id}'
                    }
                    
            # Create integration order based on dependencies
            integration_order = self._create_integration_order(subtasks)
            
            # Integrate components in order
            integrated_code = ''
            for subtask_id in integration_order:
                component = self.generated_components[subtask_id]
                integrated_code += f"\n# {subtask_id} - {component['category']}\n"
                integrated_code += component['code']
                
            # Validate integrated solution
            validation_result = self.code_validator.validate_code(
                code=integrated_code,
                category='integration'
            )
            
            if not validation_result['success']:
                return {
                    'success': False,
                    'error': f'Integration validation failed: {validation_result["errors"]}'
                }
                
            return {
                'success': True,
                'integrated_code': integrated_code,
                'validation': validation_result
            }
            
        except Exception as e:
            self.logger.error(f'Component integration failed: {e}')
            return self._create_error_response(str(e))
            
    def _create_integration_order(self, subtasks: List[SubTask]) -> List[str]:
        """Create ordered list of components for integration based on dependencies"""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(subtask):
            if subtask.id in visited:
                return
            visited.add(subtask.id)
            for dep_id in subtask.dependencies:
                dep_task = next(st for st in subtasks if st.id == dep_id)
                visit(dep_task)
            order.append(subtask.id)
            
        for subtask in subtasks:
            if subtask.id not in visited:
                visit(subtask)
                
        return order