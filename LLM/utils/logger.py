# utils/logger.py
import logging
import os

def setup_logger(name, log_file, level=logging.INFO, simple_format=False):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if simple_format:
        formatter = logging.Formatter('%(asctime)s - %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evităm adăugarea multiplă a handlerelor
    if not logger.handlers:
        logger.addHandler(file_handler)
        
    return logger