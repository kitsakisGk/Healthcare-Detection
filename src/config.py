"""
Configuration Management Module
Loads and validates configuration from YAML files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the healthcare AI system"""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration

        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """Validate required configuration fields"""
        required_sections = ['project', 'data', 'training', 'models']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., 'data.batch_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        return self.get('data.num_classes', 4)

    @property
    def class_names(self) -> list:
        """Get class names"""
        return self.get('data.class_names', ["NORMAL", "BACTERIAL", "VIRAL", "COVID19"])

    @property
    def img_size(self) -> int:
        """Get image size"""
        return self.get('data.img_size', 224)

    @property
    def batch_size(self) -> int:
        """Get batch size"""
        return self.get('data.batch_size', 32)

    @property
    def learning_rate(self) -> float:
        """Get learning rate"""
        return self.get('training.optimizer.lr', 0.0001)

    @property
    def epochs(self) -> int:
        """Get number of epochs"""
        return self.get('training.epochs', 30)

    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, classes={self.num_classes})"


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance (singleton pattern)

    Args:
        config_path: Optional path to config file

    Returns:
        Config instance
    """
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config
