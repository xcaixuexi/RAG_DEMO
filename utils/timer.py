import time
import logging
import functools

logger = logging.getLogger(__name__)

def timed(label: str = ""):
    """
    计时装饰器，自动记录函数执行耗时。
    用法：@timed("规则路由")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__qualname__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"[耗时] {name}: {elapsed:.1f}ms")
            return result
        return wrapper
    return decorator