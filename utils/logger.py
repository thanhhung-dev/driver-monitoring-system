import logging
import os
import sys
import yaml

def setup_logger(name:str, config_path:str = "config.yaml") -> logging.Logger:
    """
    Khoi Tao Logger tu file Config.yaml
    Tham So
    Name:
    config_path
    """
    log_level = logging.INFO
    log_file: str | None = None
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        log_level_str = config.get("system", {}).get("log_level", "INFO")
        log_level = getattr(logging, log_level_str, logging.INFO)
        log_file = config.get("system", {}).get("log_file")
        if log_file and not os.path.isabs(log_file):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            log_file = os.path.join(config_dir, log_file)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Tranh Spam messages nhieu lan
    if logger.handlers:
        return logger
    
    # Format message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
     # Console handler (force UTF-8 để hỗ trợ ký tự Unicode trên Windows)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
 
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
 
    return logger
