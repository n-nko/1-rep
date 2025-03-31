import os
import sys
import logging
import time
import subprocess
from threading import Lock
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
from assistant_enhanced import JARVIS
from web_learner import WebLearner

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

class TelegramBotHandler:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError('TELEGRAM_BOT_TOKEN not found in environment variables')
        
        # Create uploads directory if it doesn't exist
        self.uploads_dir = Path('uploads')
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Initialize conversation contexts with enhanced management
        self.conversation_contexts: Dict[int, list] = {}
        self.max_context_length = 5  # Keep last 5 exchanges for context
        self.context_lock = Lock()
        
        # Initialize JARVIS and WebLearner with enhanced error handling
        try:
            self.jarvis = JARVIS()
            self.web_learner = WebLearner()
            logger.info("Successfully initialized JARVIS and WebLearner")
        except Exception as e:
            logger.error(f"Failed to initialize JARVIS and WebLearner: {e}")
            raise
        
        # Enhanced state management
        self.user_states = {}
        self.conversation_timeouts = {}
        self.cleanup_interval = 1800  # 30 minutes
        self.last_cleanup = time.time()
        
    def _initialize_model(self):
        """Initialize Ollama model with retry mechanism"""
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                with self.model_lock:
                    ollama.pull(self.model)
                    logger.info(f"Successfully initialized {self.model} model")
                    return
            except Exception as e:
                last_error = e
                retry_count += 1
                wait_time = 2 ** retry_count
                logger.warning(f"Model initialization attempt {retry_count} failed: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to initialize model after {self.max_retries} attempts")
        raise last_error or Exception("Failed to initialize model")
        
    def get_time_based_greeting(self):
        hour = int(time.strftime('%H'))
        if 5 <= hour < 12:
            return "Good morning Mr. Nesterenko"
        elif 12 <= hour < 18:
            return "Good day Mr. Nesterenko"
        else:
            return "Good evening Mr. Nesterenko"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        greeting = self.get_time_based_greeting()
        user_id = update.effective_user.id
        self.user_states[user_id] = True
        
        # Create keyboard markup with JARVIS control buttons
        keyboard = [
            [InlineKeyboardButton("Start JARVIS", callback_data='start_jarvis')],
            [InlineKeyboardButton("Stop JARVIS", callback_data='stop_jarvis')],
            [InlineKeyboardButton("Restart JARVIS", callback_data='restart_jarvis')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f'{greeting}. How can I help you today?\n\nYou can control JARVIS using the buttons below:',
            reply_markup=reply_markup
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
        I can help you with various tasks. Here are some commands:
        /start - Start the conversation
        /help - Show this help message
        /clear - Clear conversation history
        /learn <url> - Learn from a website and answer questions about it
        
        You can also:
        - Send me any message and I'll respond!
        - Send photos, videos, or documents for me to learn from
        """
        await update.message.reply_text(help_text)

    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Clear user's conversation context
        user_id = update.effective_user.id
        if user_id in self.conversation_contexts:
            del self.conversation_contexts[user_id]
        await update.message.reply_text('Conversation history cleared!')

    async def learn_from_website(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /learn command to learn from a website"""
        try:
            # Extract URL from command
            args = context.args
            if not args:
                await update.message.reply_text('Please provide a URL to learn from. Usage: /learn <url>')
                return
            
            url = args[0]
            await update.message.reply_text(f'Learning from {url}...')
            
            # Use WebLearner to process the website
            result = self.web_learner.learn_from_website(url)
            
            if result['success']:
                facts = result['learned_facts']
                if facts:
                    response = "Here's what I learned:\n\n" + "\n".join(f"- {fact}" for fact in facts[:5])
                    if len(facts) > 5:
                        response += f"\n\nAnd {len(facts) - 5} more facts. You can ask me questions about what I learned!"
                else:
                    response = "I processed the website but couldn't extract any significant facts. You can still ask me questions about it!"
            else:
                response = f"Sorry, I encountered an error while learning from the website: {result['error']}"
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Failed to learn from website: {e}")
            await update.message.reply_text('Sorry, I encountered an error while processing the website. Please try again later.')
    
    async def handle_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded files (photos, videos, documents)"""
        try:
            user_id = update.effective_user.id
            message = update.message
            file_type = None
            file_obj = None

            if message.photo:
                file_type = 'photo'
                file_obj = message.photo[-1]  # Get the largest photo size
            elif message.video:
                file_type = 'video'
                file_obj = message.video
            elif message.document:
                file_type = 'document'
                file_obj = message.document

            if not file_obj:
                await message.reply_text("Sorry, I couldn't process this file type.")
                return

            # Create user-specific directory
            user_dir = self.uploads_dir / str(user_id)
            user_dir.mkdir(exist_ok=True)

            # Download file
            file = await file_obj.get_file()
            file_extension = Path(file.file_path).suffix if file.file_path else ''
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{file_type}_{timestamp}{file_extension}"
            file_path = user_dir / filename

            await file.download_to_drive(file_path)

            # Process the file based on type
            await message.reply_text(f"I've received your {file_type}. Let me analyze it...")

            # Process the file using JARVIS
            result = self.jarvis.process_file(str(file_path), file_type)

            if result['success']:
                await message.reply_text(
                    f"I've successfully processed your {file_type}. {result['message']}"
                    "\nI'll use this information to enhance our interactions."
                )
            else:
                await message.reply_text(
                    f"I had some trouble processing your {file_type}: {result['message']}"
                    "\nPlease try again or send a different file."
                )

        except Exception as e:
            logger.error(f"Error handling file upload: {e}")
            await message.reply_text(
                "Sorry, I encountered an error while processing your file. Please try again later."
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages with enhanced error handling and resource management"""
        try:
            user_id = update.effective_user.id
            if user_id not in self.user_states:
                self.user_states[user_id] = True

            user_message = update.message.text
            current_time = time.time()
            
            # Periodic cleanup of old conversations
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_conversations()
                self.last_cleanup = current_time

            # Reset conversation if timeout (30 minutes)
            if user_id in self.conversation_timeouts:
                if current_time - self.conversation_timeouts[user_id] > 1800:  # 30 minutes
                    # Reset conversation state
                    if user_id in self.conversation_contexts:
                        del self.conversation_contexts[user_id]
                    if user_id in self.user_states:
                        del self.user_states[user_id]

            self.conversation_timeouts[user_id] = current_time

            # Handle special commands or keywords
            if user_message.lower() in ['time', 'what time', 'current time']:
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                await update.message.reply_text(f'Current time is: {current_time}')
                return

            # Get conversation context for the user
            user_context = self.conversation_contexts.get(user_id, [])
            
            # Prepare context-aware prompt
            context_prompt = "\n".join([f"{'User: ' if i%2==0 else 'Assistant: '}{msg}" for i, msg in enumerate(user_context)])
            full_prompt = f"{context_prompt}\nUser: {user_message}\nAssistant:"
            
            # Generate response with JARVIS
            try:
                response = self.jarvis.process_message(user_message)
                
                # Update conversation context
                user_context.extend([user_message, response])
                if len(user_context) > self.max_context_length * 2:  # Keep context size manageable
                    user_context = user_context[-self.max_context_length * 2:]
                self.conversation_contexts[user_id] = user_context
                    
                # Split long responses
                if len(response) > 4096:  # Telegram message limit
                    chunks = [response[i:i+4096] for i in range(0, len(response), 4096)]
                    for chunk in chunks:
                        await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(response)

            except Exception as e:
                logger.error(f"Failed to generate response: {e}")
                await update.message.reply_text(
                    "I apologize, but I encountered an error generating a response. Please try again."
                )

        except Exception as e:
            error_msg = f"Error processing message: {e}"
            if isinstance(e, sqlite3.OperationalError):
                error_msg = f"Database is temporarily busy. Please try again in a moment. (Attempt {max_retries} of {max_retries})"
            logger.error(error_msg)
            await update.message.reply_text(
                "I apologize, but I encountered an error processing your message. Please try again."
            )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error(f"Update {update} caused error {context.error}")
        if update and update.message:
            await update.message.reply_text(
                "Sorry, I encountered an error. Please try again later."
            )

    async def button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks for JARVIS control with enhanced process management"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == 'start_jarvis':
                # Check if JARVIS is already running
                from restart_jarvis import find_jarvis_processes
                if find_jarvis_processes():
                    await query.edit_message_text("JARVIS is already running!")
                    return
                    
                # Start JARVIS components
                process = subprocess.Popen(
                    [sys.executable, 'restart_jarvis.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await query.edit_message_text("Starting JARVIS components...")
                
                # Wait briefly to check for immediate failures
                try:
                    process.wait(timeout=2)
                    if process.returncode != 0:
                        stderr = process.stderr.read().decode()
                        logger.error(f"Failed to start JARVIS: {stderr}")
                        await query.edit_message_text("Failed to start JARVIS. Please check the logs.")
                        return
                except subprocess.TimeoutExpired:
                    # Process is still running, which is good
                    pass
                
            elif query.data == 'stop_jarvis':
                # Find and terminate JARVIS processes
                from restart_jarvis import find_jarvis_processes, terminate_process
                processes = find_jarvis_processes()
                if not processes:
                    await query.edit_message_text("JARVIS is not currently running.")
                    return
                    
                success = True
                for proc in processes:
                    if not terminate_process(proc):
                        success = False
                        
                if success:
                    await query.edit_message_text("JARVIS has been stopped successfully.")
                else:
                    await query.edit_message_text("Some JARVIS components could not be stopped. Please check the logs.")
                
            elif query.data == 'restart_jarvis':
                # First stop existing processes
                from restart_jarvis import find_jarvis_processes, terminate_process
                processes = find_jarvis_processes()
                for proc in processes:
                    terminate_process(proc)
                
                # Start new instance
                process = subprocess.Popen(
                    [sys.executable, 'restart_jarvis.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await query.edit_message_text("Restarting JARVIS...")
                
                # Wait briefly to check for immediate failures
                try:
                    process.wait(timeout=2)
                    if process.returncode != 0:
                        stderr = process.stderr.read().decode()
                        logger.error(f"Failed to restart JARVIS: {stderr}")
                        await query.edit_message_text("Failed to restart JARVIS. Please check the logs.")
                        return
                except subprocess.TimeoutExpired:
                    # Process is still running, which is good
                    pass
                
        except Exception as e:
            logger.error(f"Error handling button click: {e}")
            await query.edit_message_text("Error occurred while processing your request. Please try again.")
            return
    
    def run(self):
        try:
            application = Application.builder().token(self.token).build()

            # Add handlers
            application.add_handler(CommandHandler("start", self.start))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("clear", self.clear_history))
            application.add_handler(CommandHandler("learn", self.learn_from_website))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            application.add_handler(CallbackQueryHandler(self.button_click))
            
            # Add file handlers
            application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO | filters.DOCUMENT, self.handle_file))
            
            # Add error handler
            application.add_error_handler(self.error_handler)

            # Start the bot
            logger.info("Starting bot...")
            application.run_polling()
            
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            raise

if __name__ == "__main__":
    bot_handler = TelegramBotHandler()
    bot_handler.run()