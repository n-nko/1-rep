import os
import sys
import time
import signal
import psutil
import logging
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='restart.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_jarvis_processes():
    """Find all running JARVIS-related processes"""
    jarvis_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any(x in str(cmdline).lower() for x in ['chat.py', 'telegram_bot.py', 'server.py']):
                jarvis_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return jarvis_processes

def cleanup_database(db_path):
    """Clean up any stale database connections and optimize the database"""
    try:
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
        logging.info(f"Database {db_path} cleaned successfully")
    except sqlite3.Error as e:
        logging.error(f"Database cleanup error for {db_path}: {e}")

def terminate_process(proc):
    """Safely terminate a process and its children"""
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        
        # Send SIGTERM to children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Send SIGTERM to parent
        parent.terminate()
        
        # Wait for processes to terminate
        gone, alive = psutil.wait_procs([parent] + children, timeout=3)
        
        # Force kill if still alive
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
        logging.info(f'Successfully terminated process {proc.pid}')
        return True
    except Exception as e:
        logging.error(f'Failed to terminate process {proc.pid}: {str(e)}')
        return False
    
def start_jarvis_components():
    """Start JARVIS components with proper error handling"""
    components = [
        {'script': 'chat.py', 'name': 'Chat Interface'},
        {'script': 'telegram_bot.py', 'name': 'Telegram Bot'},
        {'script': 'server.py', 'name': 'Web Server'}
    ]
    
    started_processes = []
    for component in components:
        try:
            process = subprocess.Popen(
                [sys.executable, component['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            started_processes.append(process)
            logging.info(f'Started {component["name"]} (PID: {process.pid})')
            time.sleep(2)  # Give each component time to initialize
        except Exception as e:
            logging.error(f'Failed to start {component["name"]}: {str(e)}')
            # Cleanup already started processes on failure
            for p in started_processes:
                terminate_process(psutil.Process(p.pid))
            return False
    return True

def main():
    try:
        # Find and terminate existing JARVIS processes
        existing_processes = find_jarvis_processes()
        if existing_processes:
            logging.info(f'Found {len(existing_processes)} existing JARVIS processes')
            for proc in existing_processes:
                terminate_process(proc)
            time.sleep(2)  # Wait for processes to fully terminate
        
        # Clean up databases
        db_paths = ["knowledge.db", "memory.db"]
        for db_path in db_paths:
            if Path(db_path).exists():
                logging.info(f"Cleaning up {db_path}...")
                cleanup_database(db_path)
        
        # Start new instances
        if start_jarvis_components():
            logging.info('JARVIS components started successfully')
            print('JARVIS has been restarted successfully!')
        else:
            logging.error('Failed to start some JARVIS components')
            print('Failed to restart JARVIS. Check restart.log for details.')
    
    except Exception as e:
        logging.error(f'Unexpected error during restart: {str(e)}')
        print('An unexpected error occurred. Check restart.log for details.')

if __name__ == '__main__':
    main()

if __name__ == "__main__":
    main()