import logging
from typing import Dict, Any, List
import ast
import re

class CodeValidator:
    def __init__(self):
        self.logger = logging.getLogger('CodeValidator')
        
    def validate_code(self, code: str, category: str) -> Dict[str, Any]:
        """Validate generated code with enhanced checks"""
        try:
            # Parse code into AST
            tree = ast.parse(code)
            
            validation = {
                'success': True,
                'warnings': [],
                'errors': [],
                'metrics': {}
            }
            
            # Basic syntax validation
            self._validate_syntax(code, validation)
            if not validation['success']:
                return validation
                
            # Code style checks
            self._check_code_style(code, validation)
            
            # Category-specific validation
            if category == 'database':
                self._validate_database_code(tree, validation)
            elif category == 'api':
                self._validate_api_code(tree, validation)
            elif category == 'interface':
                self._validate_interface_code(tree, validation)
                
            # Calculate code metrics
            self._calculate_metrics(tree, validation)
            
            return validation
            
        except Exception as e:
            self.logger.error(f'Code validation failed: {e}')
            return {
                'success': False,
                'errors': [str(e)],
                'warnings': [],
                'metrics': {}
            }
            
    def _validate_syntax(self, code: str, validation: Dict[str, Any]):
        """Validate code syntax"""
        try:
            ast.parse(code)
        except SyntaxError as e:
            validation['success'] = False
            validation['errors'].append(f'Syntax error: {str(e)}')
            
    def _check_code_style(self, code: str, validation: Dict[str, Any]):
        """Check code style against PEP 8 guidelines"""
        # Line length
        for i, line in enumerate(code.split('\n'), 1):
            if len(line.strip()) > 100:
                validation['warnings'].append(f'Line {i} exceeds 100 characters')
                
        # Bare except clauses
        if re.search(r'except\s*:', code):
            validation['warnings'].append('Bare except clause used')
            
        # Print statements
        if 'print(' in code:
            validation['warnings'].append('Print statements should be replaced with logging')
            
        # Multiple imports per line
        if re.search(r'import.*,.*', code):
            validation['warnings'].append('Multiple imports on one line')
            
    def _validate_database_code(self, tree: ast.AST, validation: Dict[str, Any]):
        """Validate database-related code"""
        has_sqlalchemy = False
        has_connection_handling = False
        has_transaction = False
        
        for node in ast.walk(tree):
            # Check for SQLAlchemy usage
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if 'sqlalchemy' in ast.unparse(node):
                    has_sqlalchemy = True
                    
            # Check for connection handling
            if isinstance(node, ast.With):
                if 'session' in ast.unparse(node) or 'connection' in ast.unparse(node):
                    has_connection_handling = True
                    
            # Check for transaction handling
            if isinstance(node, ast.Call):
                if 'commit' in ast.unparse(node) or 'rollback' in ast.unparse(node):
                    has_transaction = True
                    
        if not has_sqlalchemy:
            validation['warnings'].append('No SQLAlchemy usage detected')
        if not has_connection_handling:
            validation['warnings'].append('No proper connection handling detected')
        if not has_transaction:
            validation['warnings'].append('No transaction handling detected')
            
    def _validate_api_code(self, tree: ast.AST, validation: Dict[str, Any]):
        """Validate API-related code"""
        has_error_handling = False
        has_input_validation = False
        has_documentation = False
        
        for node in ast.walk(tree):
            # Check for error handling
            if isinstance(node, ast.Try):
                has_error_handling = True
                
            # Check for input validation
            if isinstance(node, ast.Assert) or 'validate' in ast.unparse(node):
                has_input_validation = True
                
            # Check for documentation
            if isinstance(node, ast.Str) and node.s.strip().startswith(('"""', "'''")):
                has_documentation = True
                
        if not has_error_handling:
            validation['warnings'].append('No error handling detected')
        if not has_input_validation:
            validation['warnings'].append('No input validation detected')
        if not has_documentation:
            validation['warnings'].append('No API documentation detected')
            
    def _validate_interface_code(self, tree: ast.AST, validation: Dict[str, Any]):
        """Validate interface-related code"""
        has_input_validation = False
        has_error_messages = False
        has_user_feedback = False
        
        for node in ast.walk(tree):
            # Check for input validation
            if isinstance(node, ast.Assert) or 'validate' in ast.unparse(node):
                has_input_validation = True
                
            # Check for error messages
            if isinstance(node, ast.Str) and ('error' in node.s.lower() or 'invalid' in node.s.lower()):
                has_error_messages = True
                
            # Check for user feedback
            if isinstance(node, ast.Call) and any(x in ast.unparse(node).lower() for x in ['message', 'notification', 'alert']):
                has_user_feedback = True
                
        if not has_input_validation:
            validation['warnings'].append('No input validation detected')
        if not has_error_messages:
            validation['warnings'].append('No error messages detected')
        if not has_user_feedback:
            validation['warnings'].append('No user feedback mechanisms detected')
            
    def _calculate_metrics(self, tree: ast.AST, validation: Dict[str, Any]):
        """Calculate code quality metrics"""
        metrics = {
            'num_functions': 0,
            'num_classes': 0,
            'num_imports': 0,
            'cognitive_complexity': 0
        }
        
        # Count basic elements
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['num_functions'] += 1
            elif isinstance(node, ast.ClassDef):
                metrics['num_classes'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['num_imports'] += 1
                
        # Calculate cognitive complexity
        metrics['cognitive_complexity'] = self._calculate_cognitive_complexity(tree)
        
        validation['metrics'] = metrics
        
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity of the code"""
        complexity = 0
        
        for node in ast.walk(tree):
            # Loops add complexity
            if isinstance(node, (ast.For, ast.While)):
                complexity += 1
                
            # Conditionals add complexity
            elif isinstance(node, ast.If):
                complexity += 1
                
            # Exception handling adds complexity
            elif isinstance(node, ast.Try):
                complexity += 1
                
            # Multiple boolean operations add complexity
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity