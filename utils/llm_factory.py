"""
utils/llm_factory.py — LLM 抽象工厂

职责：
    - 统一管理所有 LLM 供应商配置（智谱 / DeepSeek / OpenAI / 本地 Ollama）
    - 通过 provider 字符串一键切换，业务代码零改动
    - 相同参数的实例走缓存，避免重复初始化

用法示例：
    # .env 中配置 LLM_PROVIDER=zhipu（或 deepseek / openai / ollama）
    llm = LLMFactory.create()                   # 读取环境变量

    llm = LLMFactory.create(provider="deepseek") # 显式指定

    # 非默认模型
    llm = LLMFactory.create(provider="openai", model_name="gpt-4o")
"""

import logging
import os
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 供应商配置表
# 新增供应商只需在此添加一行，业务层无需修改
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderConfig:
    """单个供应商的静态配置"""
    default_model: str
    base_url:      str
    env_key:       str              # 从哪个环境变量读取 API Key
    description:   str = ""


_PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "zhipu": ProviderConfig(
        default_model = "glm-4.5-air",
        base_url      = "https://open.bigmodel.cn/api/paas/v4/",
        env_key       = "ZHIPU_API_KEY",
        description   = "智谱 GLM 系列",
    ),
    "deepseek": ProviderConfig(
        default_model = "deepseek-v4-flash",
        base_url      = "https://api.deepseek.com",
        env_key       = "DEEPSEEK_API_KEY",
        description   = "DeepSeek 系列",
    ),
    "openai": ProviderConfig(
        default_model = "gpt-4o-mini",
        base_url      = "https://api.openai.com/v1",
        env_key       = "OPENAI_API_KEY",
        description   = "OpenAI GPT 系列",
    ),
    "ollama": ProviderConfig(
        default_model = "qwen2.5:7b",
        base_url      = "http://localhost:11434/v1",
        env_key       = "OLLAMA_API_KEY",        # Ollama 不需要真实 key，随意填
        description   = "本地 Ollama（离线）",
    ),
}


# ─────────────────────────────────────────────
# 工厂类
# ─────────────────────────────────────────────

class LLMFactory:
    """
    LLM 工厂。
    对外只暴露两个方法：
        create()    — 返回 ChatOpenAI 实例（有缓存）
        list_providers() — 列出已注册的供应商
    """

    @staticmethod
    def create(
        provider:    Optional[str]  = None,
        model_name:  Optional[str]  = None,
        temperature: float          = 0.0,
        top_p:       float          = 0.9,
        max_tokens:  int            = 2048,
        api_key:     Optional[str]  = None,
        base_url:    Optional[str]  = None,
    ) -> ChatOpenAI:
        """
        创建（或复用缓存的）ChatOpenAI 实例。

        参数优先级（从高到低）：
            显式传参 > 环境变量 > 默认值

        Args:
            provider:    供应商标识（"zhipu" / "deepseek" / "openai" / "ollama"）
                         默认读取 LLM_PROVIDER 环境变量，再 fallback 到 "zhipu"
            model_name:  模型名称，不传时用供应商的 default_model
            temperature: 生成温度
            top_p:       核采样参数
            max_tokens:  最大输出 token 数
            api_key:     API Key（不传时从对应环境变量读取）
            base_url:    API Base URL（不传时用供应商默认值）

        Returns:
            ChatOpenAI 实例
        """
        load_dotenv()

        # 确定供应商
        resolved_provider = (
            provider
            or os.getenv("LLM_PROVIDER", "zhipu")
        ).lower()

        if resolved_provider not in _PROVIDER_REGISTRY:
            available = ", ".join(_PROVIDER_REGISTRY.keys())
            raise ValueError(
                f"未知供应商 '{resolved_provider}'，可用: {available}\n"
                f"如需新增，请在 utils/llm_factory.py 的 _PROVIDER_REGISTRY 中注册。"
            )

        cfg = _PROVIDER_REGISTRY[resolved_provider]

        # 确定各参数
        resolved_model   = model_name or os.getenv("LLM_MODEL", cfg.default_model)
        resolved_key     = api_key    or os.getenv(cfg.env_key, "")
        resolved_url     = base_url   or cfg.base_url

        # Ollama 不需要真实 key
        if resolved_provider == "ollama" and not resolved_key:
            resolved_key = "ollama"

        if not resolved_key:
            raise ValueError(
                f"缺少 API Key：请在 .env 中设置 {cfg.env_key}，"
                f"或通过 api_key 参数传入。"
            )

        # 利用 _cached_build 做参数级缓存
        return LLMFactory._cached_build(
            provider    = resolved_provider,
            model_name  = resolved_model,
            temperature = temperature,
            top_p       = top_p,
            max_tokens  = max_tokens,
            api_key     = resolved_key,
            base_url    = resolved_url,
        )

    @staticmethod
    @lru_cache(maxsize=16)
    def _cached_build(
        provider:    str,
        model_name:  str,
        temperature: float,
        top_p:       float,
        max_tokens:  int,
        api_key:     str,
        base_url:    str,
    ) -> ChatOpenAI:
        """
        内部带缓存的构造器。
        相同参数组合只初始化一次，后续复用同一实例。
        lru_cache 要求所有参数可哈希，因此用独立函数而非实例方法。
        """
        logger.info(
            f"[LLMFactory] 初始化 LLM: provider={provider} "
            f"model={model_name} temperature={temperature}"
        )
        llm = ChatOpenAI(
            model       = model_name,
            temperature = temperature,
            top_p       = top_p,
            max_tokens  = max_tokens,
            api_key     = api_key,
            base_url    = base_url,
        )
        logger.info(f"[LLMFactory] LLM 初始化完成: {provider}/{model_name}")
        return llm

    @staticmethod
    def list_providers() -> dict[str, str]:
        """返回已注册的供应商及说明，供运维/调试查看"""
        return {k: v.description for k, v in _PROVIDER_REGISTRY.items()}

    @staticmethod
    def clear_cache() -> None:
        """清空 LLM 实例缓存（测试 / 热更新 key 时使用）"""
        LLMFactory._cached_build.cache_clear()
        logger.info("[LLMFactory] 实例缓存已清空")
