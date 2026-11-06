import os
import yaml

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Đọc file cấu hình YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config