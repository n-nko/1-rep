import os
import logging
import atexit
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='server.log'
)
logger = logging.getLogger('LlamaServer')

# Load environment variables
load_dotenv()

class CodeRequest(BaseModel):
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

class CodeResponse(BaseModel):
    generated_code: str
    error: str = None

# Model initialization with enhanced error handling and cleanup
MODEL_PATH = os.getenv('MODEL_PATH', os.path.abspath(os.path.join('models', 'mistral-7b-v0.1.Q3_K_S.gguf')))
model = None
tokenizer = None

def initialize_model():
    global model, tokenizer
    try:
        logger.info(f'Loading model from {MODEL_PATH}')
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        logger.info('Model loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        raise

def cleanup_resources():
    global model, tokenizer
    try:
        if model:
            model.cpu()
            del model
        if tokenizer:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info('Resources cleaned up successfully')
    except Exception as e:
        logger.error(f'Error during cleanup: {e}')

# Register cleanup function
atexit.register(cleanup_resources)

# Initialize model
initialize_model()

# Initialize the LlamaServer
model_path = os.getenv('MODEL_PATH', os.path.abspath(os.path.join('models', 'mistral-7b-v0.1.Q3_K_S.gguf')))
llama_server = None

try:
    llama_server = LlamaServer(model_path)
    logger.info(f"Server initialized with model: {model_path}")
except Exception as e:
    logger.error(f"Failed to initialize server: {e}")

@app.post('/generate', response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    try:
        # Input validation
        if not request.prompt.strip():
            raise ValueError('Empty prompt provided')
        if request.max_length > 4096:
            raise ValueError('max_length exceeds maximum allowed value of 4096')
        if not (0.0 <= request.temperature <= 1.0):
            raise ValueError('temperature must be between 0.0 and 1.0')

        # Prepare input with timeout handling
        try:
            inputs = tokenizer(request.prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
        except Exception as e:
            logger.error(f'Failed to tokenize input: {e}')
            raise HTTPException(status_code=400, detail='Invalid input format')

        # Generate code with resource management
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=min(request.max_length, 4096),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=1,
                    repetition_penalty=1.1
                )

            # Decode output with error handling
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return CodeResponse(generated_code=generated_text)

        except torch.cuda.OutOfMemoryError:
            logger.error('GPU out of memory error')
            cleanup_resources()
            initialize_model()
            raise HTTPException(status_code=503, detail='Server temporarily unavailable. Please try again.')
        except Exception as e:
            logger.error(f'Code generation failed: {e}')
            raise HTTPException(status_code=500, detail='Internal server error during code generation')

    except ValueError as e:
        logger.warning(f'Invalid request parameters: {e}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise HTTPException(status_code=500, detail='Internal server error')

@app.get('/health')
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not Found', 'message': 'The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.'}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed', 'message': 'The requested method is not supported for this endpoint'}), 405

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)