import logging
import os
import sys
import yaml


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config with a robust fallback for Windows encoding issues."""
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "rb") as handle:
        raw_bytes = handle.read()

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            sanitized_text = "".join(ch if ch in "\t\n\r" or ch.isprintable() else " " for ch in text)
            return yaml.safe_load(sanitized_text) or {}
        except (UnicodeDecodeError, yaml.YAMLError):
            continue

    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        sanitized_text = "".join(ch if ch in "\t\n\r" or ch.isprintable() else " " for ch in text)
        return yaml.safe_load(sanitized_text) or {}
    except yaml.YAMLError:
        return {}


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
        config = load_yaml_config(config_path)
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
