import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from pathlib import Path
from dotenv import load_dotenv
from chat_interface import ChatInterface, Message, ChatPlatform

class TelegramInterface:
    def __init__(self):
        self.logger = logging.getLogger('TelegramInterface')
        
        # Load environment variables
        load_dotenv()
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError('TELEGRAM_BOT_TOKEN not found in environment variables')
            
        # Initialize chat interface
        self.chat_interface = ChatInterface()
        
        # Create uploads directory
        self.uploads_dir = Path('uploads')
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Initialize application
        self.app = Application.builder().token(self.token).build()
        
    async def setup(self):
        """Setup telegram bot handlers"""
        try:
            # Add command handlers
            self.app.add_handler(CommandHandler('start', self.start))
            self.app.add_handler(CommandHandler('help', self.help_command))
            self.app.add_handler(CommandHandler('clear', self.clear_history))
            
            # Add callback query handler
            self.app.add_handler(CallbackQueryHandler(self.button_click))
            
            # Add message handler
            self.app.add_handler(MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message
            ))
            
            self.logger.info('Telegram bot handlers setup successfully')
            
        except Exception as e:
            self.logger.error(f'Failed to setup telegram bot: {e}', exc_info=True)
            raise
            
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            # Create keyboard markup
            keyboard = [
                [InlineKeyboardButton("Help", callback_data='help')],
                [InlineKeyboardButton("Clear History", callback_data='clear')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Create welcome message
            message = Message(
                content='/start',
                platform=ChatPlatform.TELEGRAM,
                user_id=str(update.effective_user.id),
                timestamp=datetime.now()
            )
            
            # Get response from chat interface
            response = await self.chat_interface.handle_message(message)
            
            await update.message.reply_text(
                f"{response}\n\nUse the buttons below or type a message to chat:",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f'Error in start command: {e}', exc_info=True)
            error_msg = f'An error occurred: {str(e)}'
            self.logger.error(error_msg)
            await update.message.reply_text('Sorry, something went wrong. Please try again later.')
            
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        try:
            message = Message(
                content='/help',
                platform=ChatPlatform.TELEGRAM,
                user_id=str(update.effective_user.id),
                timestamp=datetime.now()
            )
            
            response = await self.chat_interface.handle_message(message)
            await update.message.reply_text(response)
            
        except Exception as e:
            self.logger.error(f'Error in help command: {e}', exc_info=True)
            error_msg = f'An error occurred: {str(e)}'
            self.logger.error(error_msg)
            await update.message.reply_text('Sorry, something went wrong. Please try again later.')
            
    async def clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        try:
            message = Message(
                content='/clear',
                platform=ChatPlatform.TELEGRAM,
                user_id=str(update.effective_user.id),
                timestamp=datetime.now()
            )
            
            response = await self.chat_interface.handle_message(message)
            await update.message.reply_text(response)
            
        except Exception as e:
            self.logger.error(f'Error in clear command: {e}', exc_info=True)
            error_msg = f'An error occurred: {str(e)}'
            self.logger.error(error_msg)
            await update.message.reply_text('Sorry, something went wrong. Please try again later.')
            
    async def button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks"""
        try:
            query = update.callback_query
            await query.answer()
            
            if query.data == 'help':
                await self.help_command(update, context)
            elif query.data == 'clear':
                await self.clear_history(update, context)
                
        except Exception as e:
            self.logger.error(f'Error in button click: {e}', exc_info=True)
            await update.callback_query.message.reply_text('Sorry, something went wrong. Please try again.')
            
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        try:
            # Create message object
            message = Message(
                content=update.message.text,
                platform=ChatPlatform.TELEGRAM,
                user_id=str(update.effective_user.id),
                timestamp=datetime.now()
            )
            
            # Get response from chat interface
            response = await self.chat_interface.handle_message(message)
            
            await update.message.reply_text(response)
            
        except Exception as e:
            self.logger.error(f'Error handling message: {e}', exc_info=True)
            error_msg = f'An error occurred: {str(e)}'
            self.logger.error(error_msg)
            await update.message.reply_text('Sorry, something went wrong. Please try again later.')
            
    async def start_polling(self):
        """Start polling for Telegram updates with enhanced error handling"""
        try:
            self.running = True
            self.logger.info("Starting Telegram Bot...")
            
            await self.setup()
            await self.app.initialize()
            await self.app.start()
            
            # Start polling in a separate task with error handling
            polling_task = asyncio.create_task(self._run_polling_with_recovery())
            self.logger.info("Telegram Bot is running!")
            
            # Keep running until stopped with improved cleanup
            try:
                while self.running:
                    if polling_task.done():
                        exc = polling_task.exception()
                        if exc:
                            self.logger.error(f"Polling task failed: {exc}")
                            # Restart polling task
                            polling_task = asyncio.create_task(self._run_polling_with_recovery())
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.logger.info("Telegram interface is shutting down...")
            finally:
                await self._cleanup_polling(polling_task)
                
        except Exception as e:
            self.logger.error(f"Failed to start polling: {e}")
            raise
            
    async def _run_polling_with_recovery(self):
        """Run polling with automatic recovery from failures"""
        while self.running:
            try:
                await self.app.run_polling()
            except Exception as e:
                self.logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)  # Wait before retry
                
    async def _cleanup_polling(self, polling_task):
        """Clean up polling task and application resources"""
        try:
            # Stop and cleanup the application first
            if hasattr(self.app, 'is_running') and self.app.is_running():
                await self.app.stop()
                await self.app.shutdown_polling()
                await self.app.shutdown()
            
            # Then cancel polling task if it's still running
            if not polling_task.done():
                polling_task.cancel()
                try:
                    await asyncio.wait_for(polling_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    self.logger.warning("Polling task force cancelled after timeout")
            
            # Properly await application shutdown
            await self.app.shutdown()
            self.logger.info("Telegram Bot shutdown completed")
            
            # Cancel remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                self.running = False
                for task in tasks:
                    task.cancel()
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    self.logger.error(f'Error during task cleanup: {e}', exc_info=True)
                finally:
                    print("Telegram Bot stopped.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
            raise
                    
        except Exception as e:
            self.logger.error(f'Failed to start Telegram polling: {e}', exc_info=True)
            print(f"\nFatal error in Telegram Interface: {str(e)}\n")
            self.running = False

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='telegram.log'
    )
    
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start telegram interface
    telegram = TelegramInterface()
    try:
        # Run the polling in the event loop
        loop.run_until_complete(telegram.start_polling())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}\n")
    finally:
        # Ensure proper cleanup
        try:
            # Cancel all running tasks
            tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
            if tasks:
                # Cancel all tasks
                for task in tasks:
                    task.cancel()
                # Wait for tasks to finish with a timeout
                try:
                    loop.run_until_complete(asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0))
                except asyncio.TimeoutError:
                    pass
            # Clean up resources
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception as e:
            print(f"\nError during cleanup: {str(e)}\n")
        finally:
            loop.close()
            sys.exit(0)

if __name__ == '__main__':
    main()