import logging  
from logging.handlers import RotatingFileHandler 
from pathlib import Path 

def setup_logging(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True) 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    
    # Console 
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    
    #File (rotate 5MB x 3 backups)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
    
    