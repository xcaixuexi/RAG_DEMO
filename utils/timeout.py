"""
utils/timeout.py — LLM 调用超时控制

设计原则：
    - 基于 concurrent.futures.ThreadPoolExecutor，兼容所有同步 LLM 调用
    - 不依赖 asyncio，与现有 LangChain 同步链完全兼容
    - 超时后主线程立即返回，子线程继续运行直至完成（守护线程模式）
    - 提供装饰器和函数两种使用方式

三道防线对应三个超时层级：
    LAYER_LLM_ROUTER   = 20s  — Supervisor._query_router()（意图分类，只需短 prompt）
    LAYER_AGENT        = 120s  — 各 Agent 的 handle()（含 SQL 生成 + DB 查询）
    LAYER_TOTAL        = 180s  — process_message() 整体兜底（防止极端情况叠加超时）

用法示例：
    # 1. 函数调用方式
    result = call_with_timeout(fn, args=(query,), kwargs={}, timeout=30, fallback={"status": "error"})

    # 2. 装饰器方式（固定超时）
    @with_timeout(seconds=30, fallback={"status": "error", ...})
    def handle(query, ...):
        ...

    # 3. 装饰器方式（不传 fallback，超时时抛 TimeoutError）
    @with_timeout(seconds=10)
    def route(...):
        ...
"""

import logging
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── 各层级超时常量（秒）────────────────────────
LAYER_LLM_ROUTER = 20    # 意图分类：短 prompt，理应很快
LAYER_AGENT      = 120   # Agent 处理：含 LLM 生成 SQL + DB 查询
LAYER_TOTAL      = 180   # 整体兜底：防止极端情况下多个环节叠加

# 复用同一个线程池，避免频繁创建销毁线程
_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="llm-timeout")


def call_with_timeout(
    fn:       Callable,
    args:     tuple        = (),
    kwargs:   dict         = None,
    timeout:  float        = LAYER_AGENT,
    fallback: Any          = None,
    label:    str          = "",
) -> Any:
    """
    以指定超时执行 fn(*args, **kwargs)。

    Args:
        fn:       要执行的可调用对象
        args:     位置参数
        kwargs:   关键字参数
        timeout:  超时时间（秒）
        fallback: 超时或出错时的返回值；为 None 时超时直接抛 TimeoutError
        label:    日志标识（调试用）

    Returns:
        fn 的返回值，或 fallback（超时/出错时）

    Raises:
        TimeoutError: 当 fallback 为 None 且超时时
    """
    kwargs = kwargs or {}
    name   = label or getattr(fn, "__name__", str(fn))

    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning(f"[超时] {name} 执行超过 {timeout}s，已中止等待")
        if fallback is not None:
            return fallback
        raise TimeoutError(f"{name} 执行超时（{timeout}s）")
    except Exception as e:
        logger.error(f"[超时控制] {name} 执行异常: {e}")
        if fallback is not None:
            return fallback
        raise


def with_timeout(
    seconds:  float,
    fallback: Any   = None,
    label:    str   = "",
):
    """
    超时装饰器工厂。

    Args:
        seconds:  超时时间（秒）
        fallback: 超时时的返回值；为 None 时超时抛 TimeoutError
        label:    日志标识，默认使用函数名

    用法：
        @with_timeout(seconds=30, fallback=_timeout_response("xxx"))
        def handle(query, ...):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        name = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return call_with_timeout(
                fn      = fn,
                args    = args,
                kwargs  = kwargs,
                timeout = seconds,
                fallback= fallback,
                label   = name,
            )
        return wrapper
    return decorator


def make_timeout_response(
    intent:  str,
    message: str = "抱歉，您的问题处理超时，请稍后重试或换个方式描述。",
    status:  str = "error",
) -> dict:
    """
    生成标准超时响应字典，与各 Agent 的响应格式保持一致。

    Args:
        intent:  意图标识，与正常响应的 intent 字段保持一致
        message: 展示给用户的提示语
        status:  响应状态，默认 "error"
    """
    return {
        "intent": intent,
        "data": {
            "message":    message,
            "total":      None,
            "list_type":  None,
            "items":      None,
            "pagination": None,
        },
        "status": status,
    }
