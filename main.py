import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from terminal_interface import TerminalInterface
from telegram_interface import TelegramInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='jarvis.log'
)
logger = logging.getLogger('JARVIS')

class JarvisLauncher:
    def __init__(self):
        self.logger = logging.getLogger('JarvisLauncher')
        load_dotenv()
        
    async def start_terminal(self):
        """Start JARVIS in terminal mode"""
        try:
            terminal = TerminalInterface()
            await terminal.start()
        except Exception as e:
            self.logger.error(f'Failed to start terminal interface: {e}', exc_info=True)
            sys.exit(1)
            
    async def start_telegram(self):
        """Start JARVIS in Telegram mode"""
        try:
            if not os.getenv('TELEGRAM_BOT_TOKEN'):
                print('Error: TELEGRAM_BOT_TOKEN not found in .env file')
                sys.exit(1)
                
            telegram = TelegramInterface()
            loop.run_until_complete(telegram.start_polling())
        except Exception as e:
            self.logger.error(f'Failed to start telegram interface: {e}', exc_info=True)
            sys.exit(1)
            
    def show_menu(self):
        """Show interface selection menu"""
        print('\nWelcome to JARVIS!')
        print('Please select an interface:')
        print('1. Terminal')
        print('2. Telegram')
        print('3. Exit')
        
        async def start(self):
            """Main entry point"""
            while True:
                self.show_menu()
                choice = input('\nEnter your choice (1-3): ').strip()
                
                if choice == '1':
                    await self.start_terminal()
                elif choice == '2':
                    await self.start_telegram()
                elif choice == '3':
                    print('Goodbye!')
                    sys.exit(0)
                else:
                    print('Invalid choice, please try again')
    async def start(self):
        """Main entry point with concurrent execution"""
        self.show_menu()
        while True:
            choice = input('\nEnter your choice (1-4): ').strip()
            
            if choice == '4':
                try:
                    await asyncio.gather(
                        self.start_terminal(),
                        self.start_telegram()
                    )
                except Exception as e:
                    self.logger.error(f'Concurrent interface error: {e}')
                    sys.exit(1)
            elif choice == '3':
                print('Goodbye!')
                sys.exit(0)
            else:
                print('Invalid choice, please try again')
    def show_menu(self):
        """Show interface selection menu"""
        print('\nWelcome to JARVIS!')
        print('Please select an interface:')
        print('1. Terminal')
        print('2. Telegram')
        print('3. Exit')
        
        while True:
            try:
                choice = input('\nEnter your choice (1-3): ').strip()
                if choice in ['1', '2', '3']:
                    return choice
                print('Invalid choice. Please enter 1, 2, or 3.')
            except KeyboardInterrupt:
                print('\nExiting...')
                sys.exit(0)
                
    async def run(self):
        """Run JARVIS with selected interfaces"""
        try:
            print('\nWelcome to JARVIS!')
            print('Select interfaces to start (you can select multiple):')
            print('1. Terminal')
            print('2. Telegram')
            print('3. Exit')
            
            while True:
                try:
                    choices = input('\nEnter your choices (e.g., "1 2" for both, or "3" to exit): ').strip().split()
                    if not choices:
                        print('Please enter at least one choice.')
                        continue
                        
                    if '3' in choices:
                        print('\nGoodbye!')
                        sys.exit(0)
                        
                    tasks = []
                    if '1' in choices:
                        tasks.append(self.start_terminal())
                    if '2' in choices:
                        tasks.append(self.start_telegram())
                        
                    if not tasks:
                        print('Invalid choices. Please enter valid numbers (1, 2, or 3).')
                        continue
                        
                    print('\nStarting selected interfaces...')
                    await asyncio.gather(*tasks)
                    break
                    
                except KeyboardInterrupt:
                    print('\nExiting...')
                    sys.exit(0)
                except ValueError:
                    print('Invalid input. Please enter numbers separated by spaces.')
                    
        except Exception as e:
            self.logger.error(f'Failed to run JARVIS: {e}', exc_info=True)
            sys.exit(1)

def main():
    launcher = JarvisLauncher()
    try:
        asyncio.run(launcher.run())
    except KeyboardInterrupt:
        print('\nExiting...')
        sys.exit(0)
    except Exception as e:
        print(f'\nFatal error: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()