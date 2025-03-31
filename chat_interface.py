import logging
import asyncio
from typing import Dict, Any, Optional, List
from queue import Queue
from threading import Lock
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='chat.log'
)
logger = logging.getLogger('ChatInterface')

class ChatPlatform(Enum):
    TERMINAL = 'terminal'
    TELEGRAM = 'telegram'

@dataclass
class Message:
    content: str
    platform: ChatPlatform
    user_id: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None

class ChatInterface:
    def __init__(self):
        self.logger = logging.getLogger('ChatInterface')
        self.message_queue = Queue()
        self.context_lock = Lock()
        self.user_contexts: Dict[str, List[Message]] = {}
        self.max_context_length = 10
        
    async def handle_message(self, message: Message) -> str:
        """Process incoming messages from any platform"""
        try:
            # Update user context
            with self.context_lock:
                if message.user_id not in self.user_contexts:
                    self.user_contexts[message.user_id] = []
                
                # Add message to context
                self.user_contexts[message.user_id].append(message)
                
                # Trim context if too long
                if len(self.user_contexts[message.user_id]) > self.max_context_length:
                    self.user_contexts[message.user_id] = self.user_contexts[message.user_id][-self.max_context_length:]
            
            # Add message to processing queue
            self.message_queue.put(message)
            
            # Process message based on platform
            if message.platform == ChatPlatform.TERMINAL:
                response = await self._handle_terminal_message(message)
            elif message.platform == ChatPlatform.TELEGRAM:
                response = await self._handle_telegram_message(message)
            else:
                raise ValueError(f'Unsupported platform: {message.platform}')
                
            return response
            
        except Exception as e:
            self.logger.error(f'Error handling message: {e}', exc_info=True)
            return f'Sorry, I encountered an error: {str(e)}'
    
    async def _handle_terminal_message(self, message: Message) -> str:
        """Handle messages from terminal interface"""
        try:
            # Process terminal-specific commands
            if message.content.startswith('/'):
                return await self._process_terminal_command(message)
            
            # Regular message processing
            return await self._process_message(message)
            
        except Exception as e:
            self.logger.error(f'Error handling terminal message: {e}', exc_info=True)
            return f'Terminal Error: {str(e)}'
    
    async def _handle_telegram_message(self, message: Message) -> str:
        """Handle messages from Telegram interface"""
        try:
            # Process Telegram-specific commands
            if message.content.startswith('/'):
                return await self._process_telegram_command(message)
            
            # Regular message processing
            return await self._process_message(message)
            
        except Exception as e:
            self.logger.error(f'Error handling Telegram message: {e}', exc_info=True)
            return f'Telegram Error: {str(e)}'
    
    async def _process_message(self, message: Message) -> str:
        """Process regular chat messages"""
        try:
            # Get user context
            context = self.user_contexts.get(message.user_id, [])
            
            # Generate response based on message and context
            response = f'Processed message: {message.content}'
            
            return response
            
        except Exception as e:
            self.logger.error(f'Error processing message: {e}', exc_info=True)
            return f'Processing Error: {str(e)}'
    
    async def _process_terminal_command(self, message: Message) -> str:
        """Process terminal-specific commands"""
        command = message.content[1:].lower()
        
        if command == 'help':
            return self._get_terminal_help()
        elif command == 'clear':
            return self._clear_context(message.user_id)
        elif command == 'exit':
            return 'Goodbye!'
        else:
            return f'Unknown command: {command}'
    
    async def _process_telegram_command(self, message: Message) -> str:
        """Process Telegram-specific commands"""
        command = message.content[1:].lower()
        
        if command == 'start':
            return self._get_telegram_welcome()
        elif command == 'help':
            return self._get_telegram_help()
        elif command == 'clear':
            return self._clear_context(message.user_id)
        else:
            return f'Unknown command: {command}'
    
    def _get_terminal_help(self) -> str:
        """Get help text for terminal interface"""
        return """
Available commands:
/help - Show this help message
/clear - Clear conversation history
/exit - Exit the chat
"""
    
    def _get_telegram_help(self) -> str:
        """Get help text for Telegram interface"""
        return """
Available commands:
/start - Start the conversation
/help - Show this help message
/clear - Clear conversation history
"""
    
    def _get_telegram_welcome(self) -> str:
        """Get welcome message for Telegram interface"""
        return "Welcome! I'm JARVIS. How can I help you today?"
    
    def _clear_context(self, user_id: str) -> str:
        """Clear conversation context for a user"""
        with self.context_lock:
            if user_id in self.user_contexts:
                self.user_contexts[user_id] = []
        return 'Conversation history cleared!'