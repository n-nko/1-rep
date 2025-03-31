import os
import time
import logging
from assistant_enhanced import JARVIS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ChatInterface')

def init_jarvis():
    """Initialize JARVIS with proper error handling and logging"""
    try:
        jarvis = JARVIS()
        print("JARVIS initialized successfully!")
        return jarvis
    except Exception as e:
        error_msg = f"Error initializing JARVIS: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return None

def chat_loop(jarvis):
    """Main chat loop for interacting with JARVIS with enhanced error handling"""
    print("\nWelcome! You can now chat with JARVIS. Type 'exit' to end the conversation.\n")
    
    context_id = str(time.time())  # Create a unique context ID for this chat session
    max_retries = 3
    retry_delay = 1.0  # Initial delay in seconds
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nJARVIS: Goodbye! Have a great day!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Process message with retry mechanism
            for attempt in range(max_retries):
                try:
                    response = jarvis.process_message(user_input, context_id)
                    if response:
                        print(f"\nJARVIS: {response}\n")
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        error_msg = f"Attempt {attempt + 1} failed: {str(e)}. Retrying..."
                        logging.warning(error_msg)
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        error_msg = f"Failed to process message after {max_retries} attempts: {str(e)}"
                        logging.error(error_msg)
                        print("\nJARVIS: I apologize, but I'm having trouble processing your message. "
                              "Please try rephrasing or try again later.\n")
            
        except KeyboardInterrupt:
            print("\n\nJARVIS: Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            error_msg = f"Unexpected error in chat loop: {str(e)}"
            logging.error(error_msg)
            print(f"\nError: An unexpected error occurred. Please try again.")
            retry_delay = 1.0  # Reset delay for next interaction

def main():
    # Initialize JARVIS
    jarvis = init_jarvis()
    if jarvis:
        try:
            # Start chat loop
            chat_loop(jarvis)
        except Exception as e:
            print(f"\nUnexpected error: {e}")
    else:
        print("\nFailed to initialize JARVIS. Please check the logs for details.")

if __name__ == "__main__":
    main()