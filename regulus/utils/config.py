"""YAML 配置加载（自 src.utils.config 迁入）。"""

from __future__ import annotations

import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Successfully loaded config from %s", config_path)
        return config
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML config: %s", e)
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置到 YAML 文件。"""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info("Config saved to %s", config_path)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """递归合并配置字典。"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
