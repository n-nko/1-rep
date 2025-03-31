import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class SubTask:
    id: str
    description: str
    category: str
    dependencies: List[str]
    status: str = 'pending'
    code: str = ''
    validation_result: Dict[str, Any] = None

class TaskCategory(Enum):
    API = 'api'
    DATABASE = 'database'
    INTERFACE = 'interface'
    INTEGRATION = 'integration'
    TESTING = 'testing'

class TaskPlanner:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('TaskPlanner')
        self.current_task = None
        self.subtasks = []
        self.execution_history = []
        
    def decompose_task(self, task_description: str) -> List[SubTask]:
        """Decompose a high-level task into subtasks using LLM"""
        # Analyze task requirements
        components = self._analyze_requirements(task_description)
        
        # Generate subtasks with dependencies
        subtasks = []
        task_id = 0
        
        # Database tasks first
        if 'database' in components:
            task_id += 1
            subtasks.append(SubTask(
                id=f'task_{task_id}',
                description='Design and implement database schema',
                category=TaskCategory.DATABASE.value,
                dependencies=[]
            ))
            
        # API tasks next
        if 'api' in components:
            task_id += 1
            subtasks.append(SubTask(
                id=f'task_{task_id}',
                description='Implement API endpoints and handlers',
                category=TaskCategory.API.value,
                dependencies=['task_1'] if 'database' in components else []
            ))
            
        # Interface tasks last
        if 'interface' in components:
            task_id += 1
            subtasks.append(SubTask(
                id=f'task_{task_id}',
                description='Implement user interface and interactions',
                category=TaskCategory.INTERFACE.value,
                dependencies=['task_2'] if 'api' in components else []
            ))
            
        # Add integration testing task
        task_id += 1
        subtasks.append(SubTask(
            id=f'task_{task_id}',
            description='Implement integration tests',
            category=TaskCategory.TESTING.value,
            dependencies=[st.id for st in subtasks]
        ))
        
        self.subtasks = subtasks
        return subtasks
    
    def _analyze_requirements(self, task_description: str) -> List[str]:
        """Analyze task description to identify required components"""
        components = []
        
        # Basic component detection
        if any(kw in task_description.lower() for kw in ['database', 'storage', 'data', 'save']):
            components.append('database')
            
        if any(kw in task_description.lower() for kw in ['api', 'endpoint', 'service']):
            components.append('api')
            
        if any(kw in task_description.lower() for kw in ['interface', 'ui', 'frontend', 'bot']):
            components.append('interface')
            
        # Ensure at least interface is included for bot tasks
        if 'bot' in task_description.lower() and 'interface' not in components:
            components.append('interface')
            
        return components
    
    def generate_code(self, subtask: SubTask) -> str:
        """Generate code for a subtask using appropriate code model with enhanced error handling and validation"""
        try:
            from code_generator import CodeGenerator, ModelType
            
            # Initialize code generator with model paths
            code_generator = CodeGenerator(
                code_llama_path='models/code-llama.gguf',
                starcoder_path='models/starcoder.gguf',
                autogpt_path='models/autogpt.gguf'
            )
            
            # Select model type based on task category and complexity
            model_type = self._select_model_type(subtask)
            
            # Get dependency code with validation
            dependency_code = self._get_dependency_code(subtask)
            
            # Generate code with enhanced error handling
            result = code_generator.generate_code(
                task_description=subtask.description,
                category=subtask.category,
                model_type=model_type,
                existing_code=dependency_code,
                max_tokens=2048
            )
            
            if result['success']:
                # Validate generated code
                validation = result.get('validation', {})
                if validation.get('security_warnings'):
                    self.logger.warning(
                        f'Security warnings for {subtask.id}: {validation["security_warnings"]}'
                    )
                return result['code']
            else:
                self.logger.error(
                    f'Code generation failed for {subtask.id}: {result.get("error")}'
                )
                return ''
            
        except Exception as e:
            self.logger.error(
                f'Code generation failed for {subtask.id}: {e}',
                exc_info=True
            )
            return ''
            
    def _select_model_type(self, subtask: SubTask) -> str:
        """Select appropriate model type based on task category and complexity"""
        if subtask.category in [TaskCategory.DATABASE.value, TaskCategory.TESTING.value]:
            return ModelType.CODE_LLAMA.value
        elif subtask.category == TaskCategory.INTERFACE.value:
            return ModelType.STARCODER.value
        else:
            # Use AutoGPT for complex tasks or when dependencies are involved
            if len(subtask.dependencies) > 2:
                return ModelType.AUTOGPT.value
            return ModelType.STARCODER.value
    
    def validate_code(self, subtask: SubTask) -> Dict[str, Any]:
        """Validate generated code for a subtask"""
        try:
            validation_results = {
                'success': False,
                'errors': [],
                'warnings': [],
                'metrics': {}
            }
            
            # Статический анализ кода
            static_analysis = self._perform_static_analysis(subtask.code)
            if not static_analysis['success']:
                validation_results['errors'].extend(static_analysis['errors'])
                return validation_results
            
            # Проверка совместимости с зависимостями
            compatibility = self._check_dependencies_compatibility(subtask)
            if not compatibility['success']:
                validation_results['errors'].extend(compatibility['errors'])
                return validation_results
            
            # Модульное тестирование с TensorFlow
            test_results = self._run_unit_tests(subtask)
            if not test_results['success']:
                validation_results['errors'].extend(test_results['errors'])
                return validation_results
                
            # Добавляем метрики TensorFlow в результаты
            if test_results.get('metrics'):
                validation_results['metrics'].update(test_results['metrics'])
                
            # Анализ производительности моделей машинного обучения
            if 'tensorflow' in subtask.code.lower():
                model_metrics = self._analyze_model_performance(subtask.code)
                validation_results['metrics']['model_performance'] = model_metrics
            
            # Анализ качества кода с использованием TensorFlow
            quality_metrics = self._analyze_code_quality(subtask.code)
            validation_results['metrics'] = quality_metrics
            
            # Проверка безопасности
            security_check = self._check_security(subtask.code)
            if not security_check['success']:
                validation_results['warnings'].extend(security_check['warnings'])
            
            validation_results['success'] = True
            return validation_results
            
        except Exception as e:
            self.logger.error(f'Code validation failed for {subtask.id}: {e}')
            return {'success': False, 'errors': [str(e)]}
    
    def integrate_components(self) -> bool:
        """Integrate all completed subtasks into final solution"""
        try:
            # Проверка готовности всех компонентов
            if not all(subtask.status == 'completed' for subtask in self.subtasks):
                self.logger.error('Not all subtasks are completed')
                return False
            
            # Создание графа зависимостей
            dependency_graph = self._build_dependency_graph()
            
            # Топологическая сортировка для определения порядка интеграции
            integration_order = self._topological_sort(dependency_graph)
            
            # Интеграция компонентов в правильном порядке
            integrated_code = ''
            for subtask_id in integration_order:
                subtask = next(st for st in self.subtasks if st.id == subtask_id)
                integrated_code = self._merge_code(integrated_code, subtask.code)
                
                # Проверка интеграции после каждого слияния
                if not self._verify_integration(integrated_code):
                    self.logger.error(f'Integration verification failed at {subtask_id}')
                    return False
            
            # Финальная оптимизация
            optimized_code = self._optimize_integrated_code(integrated_code)
            
            # Сохранение результата
            self._save_integrated_solution(optimized_code)
            
            return True
            
        except Exception as e:
            self.logger.error(f'Component integration failed: {e}')
            return False
    
    def execute_task(self, task_description: str) -> bool:
        """Execute complete task workflow"""
        try:
            # Decompose task
            self.current_task = task_description
            subtasks = self.decompose_task(task_description)
            
            # Process each subtask
            for subtask in subtasks:
                # Generate code
                subtask.code = self.generate_code(subtask)
                if not subtask.code:
                    self.logger.error(f'Failed to generate code for {subtask.id}')
                    return False
                    
                # Validate code
                subtask.validation_result = self.validate_code(subtask)
                if not subtask.validation_result.get('success'):
                    self.logger.error(f'Code validation failed for {subtask.id}')
                    return False
                    
                subtask.status = 'completed'
                self.execution_history.append({
                    'task_id': subtask.id,
                    'status': 'success',
                    'validation': subtask.validation_result
                })
                
            # Integrate components
            if not self.integrate_components():
                self.logger.error('Component integration failed')
                return False
                
            return True
            
        except Exception as e:
            error_context = {
                'task_description': task_description,
                'current_subtasks': [st.id for st in (subtasks or [])],
                'execution_history': self.execution_history
            }
            self.logger.error(f'Task execution failed: {e}', extra={'error_context': error_context})
            return False