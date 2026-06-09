"""
controller/session_manager.py — 服务端内存历史会话存储（TTL 版）

改进：
    - 每条 session 记录附带最后活跃时间戳（last_active）
    - 后台线程每 _GC_INTERVAL_SEC 秒触发一次 GC，清除超时 session
    - 默认 TTL = 2 小时，可通过环境变量 SESSION_TTL_SEC 覆盖

key 格式："{session_id}:{agent_type}"
value 结构：{"messages": deque, "last_active": float(timestamp)}

各 Agent 配置：
    chitchat          → maxlen=6   (3轮×2)
    knowledge         → maxlen=10  (5轮×2)
    job_search        → maxlen=10  (5轮×2)
    job_manage        → maxlen=10  (5轮×2)
    candidate_search  → maxlen=10  (5轮×2，只存用户提问和简短摘要)
"""

import logging
import os
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


# ── 配置 ──────────────────────────────────────
_SESSION_TTL_SEC   = int(os.getenv("SESSION_TTL_SEC",   str(2 * 3600)))   # 默认 2 小时
_GC_INTERVAL_SEC   = int(os.getenv("SESSION_GC_SEC",    str(10 * 60)))    # 默认 10 分钟 GC 一次

# 各 Agent 的 maxlen（滚动窗口大小）
_MAXLEN: dict[str, int] = {
    "chitchat":         6,     # 3 轮 × 2
    "knowledge":        10,    # 5 轮 × 2
    "job_search":       10,
    "job_manage":       10,
    "candidate_search": 10,    # 只存 message 摘要，不存列表数据
}

# ── 全局存储 ──────────────────────────────────
# 结构：{ "session_id:agent_type": {"messages": deque, "last_active": float} }
_store: dict[str, dict] = {}
_store_lock = threading.Lock()


# ─────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────

def _key(session_id: str, agent_type: str) -> str:
    return f"{session_id}:{agent_type}"


def _touch(entry: dict) -> None:
    """更新最后活跃时间"""
    entry["last_active"] = time.monotonic()


def _gc() -> None:
    """
    垃圾回收：删除所有超过 TTL 的 session 条目。
    在持锁状态下运行，线程安全。
    """
    now     = time.monotonic()
    expired = [
        k for k, v in _store.items()
        if now - v["last_active"] > _SESSION_TTL_SEC
    ]
    for k in expired:
        del _store[k]
    if expired:
        logger.info(f"[SessionManager] GC 清理 {len(expired)} 条过期 session")


# ─────────────────────────────────────────────
# 后台 GC 线程
# ─────────────────────────────────────────────

def _gc_loop() -> None:
    """后台定期 GC，守护线程，随主进程退出"""
    while True:
        time.sleep(_GC_INTERVAL_SEC)
        try:
            with _store_lock:
                _gc()
        except Exception as e:
            logger.error(f"[SessionManager] GC 异常: {e}")


_gc_thread = threading.Thread(target=_gc_loop, daemon=True, name="session-gc")
_gc_thread.start()
logger.debug(f"[SessionManager] 后台 GC 已启动，间隔={_GC_INTERVAL_SEC}s TTL={_SESSION_TTL_SEC}s")


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def get_history(session_id: str, agent_type: str) -> list[dict]:
    """
    读取指定 session + agent 的历史，返回 list[dict]。
    不存在或已过期时返回空列表。
    """
    k = _key(session_id, agent_type)
    with _store_lock:
        entry = _store.get(k)
        if entry is None:
            return []
        _touch(entry)
        return list(entry["messages"])


def append_history(
    session_id: str,
    agent_type: str,
    query:      str,
    reply:      str,
) -> None:
    """
    追加一轮对话到历史。
    不存在时按 agent_type 配置的 maxlen 创建。

    Args:
        session_id: 前端传入的会话 ID
        agent_type: "chitchat" | "knowledge" | "job_search" | "job_manage" | "candidate_search"
        query:      用户原始输入
        reply:      Agent 回复的 message 字段内容
    """
    k = _key(session_id, agent_type)
    with _store_lock:
        if k not in _store:
            maxlen = _MAXLEN.get(agent_type, 6)
            _store[k] = {
                "messages":    deque(maxlen=maxlen),
                "last_active": time.monotonic(),
            }
        entry = _store[k]
        entry["messages"].append({"role": "user",      "content": query})
        entry["messages"].append({"role": "assistant", "content": reply})
        _touch(entry)


def clear_session(session_id: str) -> None:
    """清除某个 session 下所有 agent 的历史"""
    prefix = f"{session_id}:"
    with _store_lock:
        keys_to_del = [k for k in _store if k.startswith(prefix)]
        for k in keys_to_del:
            del _store[k]
    logger.info(f"[SessionManager] 已清除 session={session_id} 全部历史")


def clear_agent(session_id: str, agent_type: str) -> None:
    """清除某个 session 下指定 agent 的历史"""
    k = _key(session_id, agent_type)
    with _store_lock:
        _store.pop(k, None)


def stats() -> dict:
    """返回当前存储状态，供监控/调试使用"""
    with _store_lock:
        total_sessions = len({k.split(":")[0] for k in _store})
        total_entries  = len(_store)
        return {
            "total_sessions": total_sessions,
            "total_entries":  total_entries,
            "ttl_sec":        _SESSION_TTL_SEC,
            "gc_interval_sec":_GC_INTERVAL_SEC,
        }
