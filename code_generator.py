import logging
from typing import Dict, Any, Optional
from enum import Enum
from llama_cpp import Llama

class ModelType(Enum):
    CODE_LLAMA = 'code_llama'
    STARCODER = 'starcoder'
    AUTOGPT = 'autogpt'

class CodeGenerator:
    def __init__(self, code_llama_path: str, starcoder_path: Optional[str] = None, autogpt_path: Optional[str] = None):
        self.logger = logging.getLogger('CodeGenerator')
        
        # Initialize Code Llama
        try:
            self.code_llama = Llama(
                model_path=code_llama_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                n_batch=512,
                embedding=True
            )
            self.logger.info('Code Llama initialized successfully')
        except Exception as e:
            self.logger.error(f'Failed to initialize Code Llama: {e}')
            raise
            
        # Initialize StarCoder if path provided
        self.starcoder = None
        if starcoder_path:
            try:
                self.starcoder = Llama(
                    model_path=starcoder_path,
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    n_batch=512,
                    embedding=True
                )
                self.logger.info('StarCoder initialized successfully')
            except Exception as e:
                self.logger.warning(f'Failed to initialize StarCoder: {e}')
                
        # Initialize AutoGPT if path provided
        self.autogpt = None
        if autogpt_path:
            try:
                self.autogpt = Llama(
                    model_path=autogpt_path,
                    n_gpu_layers=-1,
                    n_ctx=4096,
                    n_batch=512,
                    embedding=True
                )
                self.logger.info('AutoGPT initialized successfully')
            except Exception as e:
                self.logger.warning(f'Failed to initialize AutoGPT: {e}')
                
    def generate_code(self, 
                     task_description: str,
                     category: str,
                     model_type: ModelType = ModelType.CODE_LLAMA,
                     existing_code: str = '',
                     max_tokens: int = 2048) -> Dict[str, Any]:
        """Generate code for a given task using specified model with enhanced error handling and validation"""
        try:
            # Input validation
            if not task_description.strip():
                raise ValueError('Task description cannot be empty')
            if max_tokens > 4096:
                raise ValueError('max_tokens exceeds maximum allowed value of 4096')
            
            # Select model based on type with enhanced error handling
            model = None
            if model_type == ModelType.CODE_LLAMA:
                model = self.code_llama
            elif model_type == ModelType.STARCODER:
                model = self.starcoder
            elif model_type == ModelType.AUTOGPT:
                model = self.autogpt
            else:
                raise ValueError(f'Invalid model type: {model_type.value}')
                
            if not model:
                raise ValueError(f'Model {model_type.value} not initialized')
            
            # Enhance prompt based on category with security considerations
            prompt = self._create_prompt(task_description, category, existing_code)
            
            # Generate code with enhanced parameters and resource management
            try:
                response = model.create_completion(
                    prompt,
                    max_tokens=min(max_tokens, 4096),
                    temperature=0.4,  # Lower temperature for more focused code generation
                    top_p=0.95,
                    top_k=50,
                    repeat_penalty=1.1,
                    stop=['```', '###', 'def test_', 'import os.path', 'import subprocess', 'exec(', 'eval(']
                )
                
                generated_code = response['choices'][0]['text'].strip()
                
                # Enhanced code validation with security checks
                validation_result = self._validate_generated_code(generated_code, category)
                
                # Check for potential security issues
                security_issues = self._check_security_issues(generated_code)
                if security_issues:
                    validation_result['security_warnings'] = security_issues
                
                return {
                    'success': True,
                    'code': generated_code,
                    'validation': validation_result
                }
                
            except Exception as e:
                self.logger.error(f'Code generation failed: {e}')
                return {
                    'success': False,
                    'error': f'Code generation failed: {str(e)}'
                }
            
        except Exception as e:
            self.logger.error(f'Code generation failed: {e}')
            return {
                'success': False,
                'error': str(e)
            }
            
    def _create_prompt(self, task_description: str, category: str, existing_code: str) -> str:
        """Create an enhanced prompt for code generation"""
        # Base prompt template
        prompt = f"""Generate Python code for the following task:

Task Description:
{task_description}

Category: {category}

Requirements:
1. Follow Python best practices and PEP 8 style guide
2. Include proper error handling
3. Add descriptive comments
4. Ensure code is modular and maintainable
"""

        # Add category-specific requirements
        if category == 'database':
            prompt += """
5. Use SQLAlchemy for database operations
6. Implement proper connection handling
7. Include database migration support
"""
        elif category == 'api':
            prompt += """
5. Follow RESTful API principles
6. Include proper input validation
7. Implement error responses
8. Add API documentation
"""
        elif category == 'interface':
            prompt += """
5. Implement clean user interface
6. Add input validation
7. Include error messages
8. Ensure responsive design
"""
            
        # Add existing code context if provided
        if existing_code:
            prompt += f"""

Existing Code:
```python
{existing_code}
```

Extend or modify the existing code to implement the required functionality.
"""
            
        prompt += "\nGenerate the code:"
        
        return prompt
        
    def _check_common_issues(self, code: str) -> List[str]:
        """Check for common code issues"""
        issues = []
        
        # Check for bare except clauses
        if 'except:' in code:
            issues.append('Bare except clause used')
            
        # Check for print statements
        if 'print(' in code:
            issues.append('Print statements should be replaced with logging')
            
        # Check for TODO comments
        if 'TODO' in code:
            issues.append('TODO comments present in code')
            
        # Check for proper docstrings
        if not any(pattern in code for pattern in ['"""', "'''"]):
            issues.append('Missing docstrings')
            
        # Check for type hints
        if not any(pattern in code for pattern in [': str', ': int', ': bool', ': List', ': Dict']):
            issues.append('Missing type hints')
            
        return issues
        
    def _calculate_code_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        import ast
        metrics = {}
        
        try:
            tree = ast.parse(code)
            
            # Calculate cyclomatic complexity
            metrics['cyclomatic_complexity'] = sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try))
            )
            
            # Calculate lines of code
            metrics['lines_of_code'] = len(code.splitlines())
            
            # Calculate comment ratio
            comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith('#'))
            metrics['comment_ratio'] = round(comment_lines / metrics['lines_of_code'] if metrics['lines_of_code'] > 0 else 0, 2)
            
            # Calculate function complexity
            metrics['function_count'] = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            metrics['class_count'] = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            
            # Calculate cognitive complexity
            metrics['cognitive_complexity'] = self._calculate_cognitive_complexity(tree)
            
        except Exception as e:
            self.logger.warning(f'Failed to calculate some metrics: {e}')
            
        return metrics
        
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity of the code"""
        complexity = 0
        
        for node in ast.walk(tree):
            # Increment for control flow statements
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            # Additional complexity for nested control flow
            if isinstance(node, ast.Try):
                complexity += len(node.handlers) + len(node.finalbody)
            # Complexity for boolean operations
            if isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
        
    def _validate_generated_code(self, code: str, category: str) -> Dict[str, Any]:
        """Perform enhanced validation of generated code"""
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            
            validation = {
                'syntax_valid': True,
                'warnings': [],
                'metrics': {}
            }
            
            # Check for common issues
            validation['warnings'].extend(self._check_common_issues(code))
            
            # Category-specific validation
            if category == 'database':
                if 'commit()' not in code:
                    validation['warnings'].append('Missing transaction commit')
                if 'rollback()' not in code:
                    validation['warnings'].append('Missing transaction rollback')
                if not any(pattern in code for pattern in ['with session:', 'Session()', 'sessionmaker']):
                    validation['warnings'].append('No proper session management found')
                    
            elif category == 'api':
                if 'try:' not in code:
                    validation['warnings'].append('Missing error handling')
                if not any(pattern in code for pattern in ['@app.route', '@router.', 'FastAPI', 'Flask']):
                    validation['warnings'].append('No API framework found')
                if 'validate' not in code.lower():
                    validation['warnings'].append('No input validation found')
                    
            elif category == 'interface':
                if not any(pattern in code for pattern in ['class', 'def __init__']):
                    validation['warnings'].append('No class definition found')
                if not any(pattern in code for pattern in ['try:', 'except']):
                    validation['warnings'].append('Missing error handling')
                if not any(pattern in code.lower() for pattern in ['validate', 'check', 'verify']):
                    validation['warnings'].append('No input validation found')
            
            # Calculate code metrics
            validation['metrics'] = self._calculate_code_metrics(code)
            
            return validation
            
        except SyntaxError as e:
            return {
                'syntax_valid': False,
                'error': str(e),
                'line_number': e.lineno,
                'offset': e.offset,
                'text': e.text
            }
            
    def improve_code(self, code: str, validation_result: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Improve code based on validation results with enhanced context awareness"""
        try:
            if not validation_result.get('syntax_valid', True):
                # Generate improved code with syntax fixes
                prompt = f"""Fix the syntax errors in the following code:

Code with errors:
```python
{code}
```

Error details:
- Line {validation_result.get('line_number')}: {validation_result.get('error')}
- Context: {validation_result.get('text')}

Provide the corrected code with proper error handling:"""
                
                response = self.code_llama.create_completion(
                    prompt,
                    max_tokens=len(code) + 512,
                    temperature=0.3,
                    top_p=0.95,
                    repeat_penalty=1.1
                )
                
                return response['choices'][0]['text'].strip()
                
            elif validation_result.get('warnings') or validation_result.get('security_warnings'):
                # Combine all warnings
                all_warnings = validation_result.get('warnings', [])
                all_warnings.extend(validation_result.get('security_warnings', []))
                
                # Improve code based on warnings with context
                prompt = f"""Improve the following code by addressing these issues:

Current code:
```python
{code}
```

Issues to fix:
{chr(10).join(f'- {w}' for w in all_warnings)}

Code metrics:
{json.dumps(validation_result.get('metrics', {}), indent=2)}

Additional context:
{json.dumps(context, indent=2) if context else 'No additional context'}

Provide improved code that addresses all issues while maintaining functionality:"""
                
                response = self.code_llama.create_completion(
                    prompt,
                    max_tokens=len(code) + 512,
                    temperature=0.3,
                    top_p=0.95,
                    repeat_penalty=1.1
                )
                
                improved_code = response['choices'][0]['text'].strip()
                
                # Validate improved code
                new_validation = self._validate_generated_code(improved_code, context.get('category') if context else None)
                if new_validation.get('syntax_valid', False) and not new_validation.get('warnings', []):
                    return improved_code
                
                self.logger.warning('Improved code still has issues, returning original code')
                return code
                
            return code
            
        except Exception as e:
            self.logger.error(f'Code improvement failed: {e}', exc_info=True)
            return code