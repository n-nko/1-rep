import logging
import sys
from code_generator import CodeGenerator, ModelType
from model_validator import ModelValidator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('model_test.log')
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger('ModelTest')
    
    # Initialize paths for all models
    code_llama_path = 'models/mistral-7b-v0.1.Q3_K_S.gguf'
    starcoder_path = 'models/mistral-7b-v0.1.Q3_K_S.gguf'  # Using same model for testing
    autogpt_path = 'models/mistral-7b-v0.1.Q3_K_S.gguf'    # Using same model for testing
    
    try:
        # Initialize CodeGenerator
        code_generator = CodeGenerator(code_llama_path, starcoder_path, autogpt_path)
        
        # Initialize ModelValidator
        validator = ModelValidator(code_generator)
        
        # Run validation tests
        logger.info('Starting model validation tests...')
        results = validator.validate_models()
        
        # Process results
        for model, result in results.items():
            logger.info(f'\n{model.upper()} Validation Results:')
            logger.info(f'Status: {result["status"]}')
            
            if result['status'] == 'error':
                logger.error(f'Error: {result.get("error")}')
            elif result['status'] == 'operational':
                logger.info('Validation successful')
                if 'validation' in result:
                    logger.info(f'Validation details: {result["validation"]}')
            elif result['status'] == 'not_initialized':
                logger.warning(f'Model not initialized: {result.get("error")}')
            elif result['status'] == 'not_available':
                logger.warning(result.get('error', 'Model not available'))
                
    except Exception as e:
        logger.error(f'Test execution failed: {e}')
        sys.exit(1)
        
    logger.info('Model validation tests completed')

if __name__ == '__main__':
    main()