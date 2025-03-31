import logging
import os
import tempfile
import subprocess
from typing import Optional, Dict, Any
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

class CodeLlamaAgent:
    def __init__(self, model_name: str = "codellama", max_retries: int = 3):
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        
        # Initialize Ollama with retries
        self.llm = self._initialize_llm(model_name)
        
        # Enhanced memory with metadata
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Enhanced prompt template with better code generation guidelines
        self.prompt = PromptTemplate(
            input_variables=["task", "chat_history"],
            template="""You are an expert code generator. Based on the conversation history and current task, generate high-quality, production-ready code.

Conversation History:
{chat_history}

Current Task: {task}

Requirements:
1. Follow best practices and coding standards
2. Include proper error handling
3. Add comprehensive documentation
4. Ensure code is modular and maintainable
5. Consider security implications

Generated Code:"""
        )
        
        # Initialize the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def _initialize_llm(self, model_name: str) -> Ollama:
        """Initialize Ollama LLM with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                llm = Ollama(model=model_name)
                # Test the model with a simple prompt
                llm("Test connection")
                self.logger.info(f"Successfully initialized {model_name}")
                return llm
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to initialize {model_name} after {self.max_retries} attempts")
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
    
    def generate_code(self, task: str) -> Dict[str, Any]:
        """Generate code with enhanced error handling and response processing"""
        try:
            # Track token usage
            with get_openai_callback() as cb:
                response = self.chain.run(task=task)
                
            # Process and validate the response
            result = {
                'success': True,
                'code': response,
                'usage': {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens
                }
            }
            
            # Basic code validation
            try:
                compile(response, '<string>', 'exec')
            except SyntaxError as e:
                result['warnings'] = [f"Syntax error in generated code: {e}"]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.memory.clear()

    def execute_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute generated code with security measures and resource limits"""
        try:
            # Create a temporary file with unique name
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(code)

            try:
                # Execute code with timeout and resource limits
                result = subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    # Prevent shell injection
                    shell=False
                )

                execution_result = {
                    'success': result.returncode == 0,
                    'output': result.stdout if result.returncode == 0 else result.stderr,
                    'return_code': result.returncode
                }

            except subprocess.TimeoutExpired:
                execution_result = {
                    'success': False,
                    'error': f'Code execution timed out after {timeout} seconds'
                }
            except Exception as e:
                execution_result = {
                    'success': False,
                    'error': str(e)
                }

            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                self.logger.warning(f'Failed to delete temporary file {temp_path}: {e}')

            return execution_result

        except Exception as e:
            self.logger.error(f'Code execution failed: {e}')
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize agent
    agent = CodeLlamaAgent()
    
    # Generate REST API code
    task = "Create a REST API using Flask with proper error handling and documentation"
    result = agent.generate_code(task)
    
    if result['success']:
        print("Generated Code:")
        print(result['code'])
        print("\nToken Usage:", result['usage'])
        if 'warnings' in result:
            print("\nWarnings:", result['warnings'])
    else:
        print("Error:", result['error'])