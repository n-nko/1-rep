import asyncio
import sys
import logging
from datetime import datetime
from typing import Optional
from chat_interface import ChatInterface, Message, ChatPlatform

class TerminalInterface:
    def __init__(self):
        self.logger = logging.getLogger('TerminalInterface')
        self.chat_interface = ChatInterface()
        self.running = False
        
    async def start(self):
        """Start the terminal interface"""
        try:
            self.running = True
            print("Welcome to JARVIS Terminal Interface!")
            print("Type '/help' for available commands or '/exit' to quit.")
            
            while self.running:
                try:
                    # Get user input
                    user_input = await self._get_input()
                    
                    if not user_input.strip():
                        continue
                        
                    # Create message object
                    message = Message(
                        content=user_input,
                        platform=ChatPlatform.TERMINAL,
                        user_id='terminal_user',
                        timestamp=datetime.now()
                    )
                    
                    # Handle exit command
                    if user_input.lower() == '/exit':
                        self.running = False
                        print("\nGoodbye from Terminal Interface!")
                        break
                        
                    # Process message
                    try:
                        response = await self.chat_interface.handle_message(message)
                        print(f"\nJARVIS: {response}\n")
                    except Exception as chat_error:
                        self.logger.error(f'Error processing message: {chat_error}', exc_info=True)
                        print(f"\nError processing message: {str(chat_error)}\n")
                        continue
                    
                except asyncio.CancelledError:
                    self.running = False
                    print("\nTerminal interface is shutting down...")
                    break
                except KeyboardInterrupt:
                    self.running = False
                    print("\nTerminal interface interrupted...")
                    break
                except Exception as e:
                    self.logger.error(f'Error processing input: {e}', exc_info=True)
                    print(f"\nError: {str(e)}\n")
                    continue
            
        except Exception as e:
            self.logger.error(f'Terminal interface error: {e}', exc_info=True)
            print(f"\nFatal error in Terminal Interface: {str(e)}\n")
            self.running = False
            
    async def _get_input(self) -> str:
        """Get input from user asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: input("You: ")
        )

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='terminal.log'
    )
    
    # Start terminal interface
    terminal = TerminalInterface()
    try:
        asyncio.run(terminal.start())
    except Exception as e:
        print(f"\nFatal error: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()