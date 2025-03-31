import logging
from typing import Dict, Any, Optional
from code_generator import CodeGenerator, ModelType

class ModelValidator:
    def __init__(self, code_generator: CodeGenerator):
        self.logger = logging.getLogger('ModelValidator')
        self.code_generator = code_generator
        
    def validate_models(self) -> Dict[str, Any]:
        """Validate the operability of Code Llama, StarCoder, and AutoGPT models"""
        results = {
            'code_llama': self._validate_code_llama(),
            'starcoder': self._validate_starcoder(),
            'autogpt': self._validate_autogpt(),
            'cooperation': self._validate_model_cooperation()
        }
        return results
        
    def _validate_code_llama(self) -> Dict[str, Any]:
        """Validate Code Llama's basic functionality"""
        try:
            # Test basic code generation
            result = self.code_generator.generate_code(
                task_description='Create a function that adds two numbers',
                category='basic',
                model_type=ModelType.CODE_LLAMA
            )
            
            if not result['success']:
                return {
                    'status': 'error',
                    'error': f"Code generation failed: {result.get('error')}"
                }
                
            # Validate the generated code
            validation = self.code_generator._validate_generated_code(result['code'], 'basic')
            if not validation.get('syntax_valid', False):
                return {
                    'status': 'error',
                    'error': f"Generated code has syntax errors: {validation.get('error')}"
                }
                
            return {
                'status': 'operational',
                'validation': validation
            }
            
        except Exception as e:
            self.logger.error(f'Code Llama validation failed: {e}')
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _validate_starcoder(self) -> Dict[str, Any]:
        """Validate StarCoder's basic functionality"""
        if not self.code_generator.starcoder:
            return {
                'status': 'not_initialized',
                'error': 'StarCoder model is not initialized'
            }
            
        try:
            # Test basic code generation
            result = self.code_generator.generate_code(
                task_description='Create a function that multiplies two numbers',
                category='basic',
                model_type=ModelType.STARCODER
            )
            
            if not result['success']:
                return {
                    'status': 'error',
                    'error': f"Code generation failed: {result.get('error')}"
                }
                
            # Validate the generated code
            validation = self.code_generator._validate_generated_code(result['code'], 'basic')
            if not validation.get('syntax_valid', False):
                return {
                    'status': 'error',
                    'error': f"Generated code has syntax errors: {validation.get('error')}"
                }
                
            return {
                'status': 'operational',
                'validation': validation
            }
            
        except Exception as e:
            self.logger.error(f'StarCoder validation failed: {e}')
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _validate_autogpt(self) -> Dict[str, Any]:
        """Validate AutoGPT's basic functionality"""
        if not self.code_generator.autogpt:
            return {
                'status': 'not_initialized',
                'error': 'AutoGPT model is not initialized'
            }
            
        try:
            # Test basic code generation with task decomposition
            result = self.code_generator.generate_code(
                task_description='Create a function that calculates factorial with error handling',
                category='basic',
                model_type=ModelType.AUTOGPT
            )
            
            if not result['success']:
                return {
                    'status': 'error',
                    'error': f"Code generation failed: {result.get('error')}"
                }
                
            # Validate the generated code
            validation = self.code_generator._validate_generated_code(result['code'], 'basic')
            if not validation.get('syntax_valid', False):
                return {
                    'status': 'error',
                    'error': f"Generated code has syntax errors: {validation.get('error')}"
                }
                
            return {
                'status': 'operational',
                'validation': validation
            }
            
        except Exception as e:
            self.logger.error(f'AutoGPT validation failed: {e}')
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def _validate_model_cooperation(self) -> Dict[str, Any]:
        """Validate cooperation between Code Llama, StarCoder, and AutoGPT"""
        if not self.code_generator.starcoder or not self.code_generator.autogpt:
            return {
                'status': 'not_available',
                'error': 'StarCoder or AutoGPT is not initialized, cooperation testing not possible'
            }
            
        try:
            # Test sequential code generation and improvement
            llama_result = self.code_generator.generate_code(
                task_description='Create a basic calculator class',
                category='basic',
                model_type=ModelType.CODE_LLAMA
            )
            
            if not llama_result['success']:
                return {
                    'status': 'error',
                    'error': f"Code Llama generation failed: {llama_result.get('error')}"
                }
                
            # Use StarCoder to improve the code with error handling
            starcoder_result = self.code_generator.generate_code(
                task_description='Improve the calculator class with error handling',
                category='basic',
                model_type=ModelType.STARCODER,
                existing_code=llama_result['code']
            )
            
            if not starcoder_result['success']:
                return {
                    'status': 'error',
                    'error': f"StarCoder improvement failed: {starcoder_result.get('error')}"
                }
                
            # Use AutoGPT to add advanced features and documentation
            autogpt_result = self.code_generator.generate_code(
                task_description='Enhance calculator with scientific functions and comprehensive documentation',
                category='basic',
                model_type=ModelType.AUTOGPT,
                existing_code=starcoder_result['code']
            )
            
            if not autogpt_result['success']:
                return {
                    'status': 'error',
                    'error': f"AutoGPT enhancement failed: {autogpt_result.get('error')}"
                }
                
            return {
                'status': 'operational',
                'llama_validation': llama_result['validation'],
                'starcoder_validation': starcoder_result['validation'],
                'autogpt_validation': autogpt_result['validation']
            }
            
        except Exception as e:
            self.logger.error(f'Model cooperation validation failed: {e}')
            return {
                'status': 'error',
                'error': str(e)
            }