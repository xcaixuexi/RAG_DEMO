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

存储两类数据，key 前缀不同：
    对话历史  "{session_id}:hist:{agent_type}"
              value: {"messages": deque, "last_active": float}

    分页缓存  "{session_id}:page:{result_type}"
              result_type: "jobs" | "candidates"
              value: {"items": list, "total_db": int, "query": str, "last_active": float}

GC 统一清理两类数据，TTL 相同。
"""

import logging
import os
import threading
import time
from collections import deque
from math import ceil
from typing import Optional

logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────
_SESSION_TTL_SEC = int(os.getenv("SESSION_TTL_SEC", str(2 * 3600)))
_GC_INTERVAL_SEC = int(os.getenv("SESSION_GC_SEC",  str(10 * 60)))

MAX_FETCH    = int(os.getenv("RESULT_MAX_FETCH",   "100"))   # SQL 层硬上限
DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", "20"))

# 各 Agent 的 maxlen（滚动窗口大小）
_HIST_MAXLEN: dict[str, int] = {
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

def _hist_key(session_id: str, agent_type: str) -> str:
    return f"{session_id}:hist:{agent_type}"

def _page_key(session_id: str, result_type: str) -> str:
    return f"{session_id}:page:{result_type}"

def _touch(entry: dict) -> None:
    """更新最后活跃时间"""
    entry["last_active"] = time.monotonic()

def _gc() -> None:
    """
    垃圾回收：删除所有超过 TTL 的 session 条目。
    在持锁状态下运行，线程安全。
    """
    now     = time.monotonic()
    expired = [k for k, v in _store.items()
               if now - v["last_active"] > _SESSION_TTL_SEC]
    for k in expired:
        del _store[k]
    if expired:
        logger.info(f"[SessionManager] GC 清理 {len(expired)} 条过期记录")


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
# 对话历史接口（原有）
# ─────────────────────────────────────────────

def get_history(session_id: str, agent_type: str) -> list[dict]:
    """
    读取指定 session + agent 的历史，返回 list[dict]。
    不存在或已过期时返回空列表。
    """
    k = _hist_key(session_id, agent_type)
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
    k = _hist_key(session_id, agent_type)
    with _store_lock:
        if k not in _store:
            maxlen = _HIST_MAXLEN.get(agent_type, 6)
            _store[k] = {
                "messages": deque(maxlen=maxlen),
                "last_active": time.monotonic(),
            }
        entry = _store[k]
        entry["messages"].append({"role": "user",      "content": query})
        entry["messages"].append({"role": "assistant", "content": reply})
        _touch(entry)


# ─────────────────────────────────────────────
# 分页缓存接口（新增）
# ─────────────────────────────────────────────

def save_page_cache(
    session_id:   str,
    result_type:  str,        # "jobs" | "candidates"
    items:        list[dict],
    total_db:     int,        # count_sql 查出的数据库真实总数
    query:        str,        # 用户原始查询，存着方便调试
) -> None:
    """
    缓存一次查询的全部结果（最多 MAX_FETCH 条）。
    每次新查询覆盖旧缓存，翻页时直接切片，不再查 DB。
    """
    k = _page_key(session_id, result_type)
    with _store_lock:
        _store[k] = {
            "items":       items,
            "total_db":    total_db,
            "query":       query,
            "last_active": time.monotonic(),
        }
    logger.info(
        f"[SessionManager] 分页缓存写入: session={session_id} "
        f"type={result_type} fetched={len(items)} total_db={total_db}"
    )


def get_page(
    session_id:  str,
    result_type: str,
    page:        int = 1,
    page_size:   int = DEFAULT_PAGE_SIZE,
) -> Optional[dict]:
    """
    从缓存中取指定页的数据。

    Returns:
        {
            "items":       [...],   # 当页数据
            "page":        1,
            "page_size":   20,
            "total_pages": 5,       # 基于 fetched 条数算出
            "fetched":     87,      # 实际取回条数（≤ MAX_FETCH）
            "total_db":    1523,    # 数据库真实总数（告知用户完整规模）
            "query":       "...",
        }
        None 表示缓存不存在或已过期（让前端重新发起对话查询）
    """
    k = _page_key(session_id, result_type)
    with _store_lock:
        entry = _store.get(k)
        if entry is None:
            return None
        _touch(entry)

        items     = entry["items"]
        total_db  = entry["total_db"]
        fetched   = len(items)
        page_size = max(1, page_size)
        total_pages = ceil(fetched / page_size) if fetched else 1
        page        = max(1, min(page, total_pages))   # 防越界

        start = (page - 1) * page_size
        end   = start + page_size

        return {
            "items":       items[start:end],
            "page":        page,
            "page_size":   page_size,
            "total_pages": total_pages,
            "fetched":     fetched,
            "total_db":    total_db,
            "query":       entry["query"],
        }


def clear_page_cache(session_id: str, result_type: Optional[str] = None) -> None:
    """清除分页缓存，result_type=None 时清除该 session 的所有分页缓存"""
    with _store_lock:
        if result_type:
            _store.pop(_page_key(session_id, result_type), None)
        else:
            prefix = f"{session_id}:page:"
            for k in [k for k in _store if k.startswith(prefix)]:
                del _store[k]


# ─────────────────────────────────────────────
# 通用清理
# ─────────────────────────────────────────────

def clear_session(session_id: str) -> None:
    """清除某个 session 下所有 agent 的历史"""
    prefix = f"{session_id}:"
    with _store_lock:
        for k in [k for k in _store if k.startswith(prefix)]:
            del _store[k]
    logger.info(f"[SessionManager] 已清除 session={session_id} 全部数据")


def clear_agent(session_id: str, agent_type: str) -> None:
    """清除某个 session 下指定 agent 的历史"""
    k = _hist_key(session_id, agent_type)
    with _store_lock:
        _store.pop(k, None)


def stats() -> dict:
    """返回当前存储状态，供监控/调试使用"""
    with _store_lock:
        keys = list(_store.keys())
    sessions   = {k.split(":")[0] for k in keys}
    hist_keys  = [k for k in keys if ":hist:" in k]
    page_keys  = [k for k in keys if ":page:" in k]
    return {
        "total_sessions":   len(sessions),
        "history_entries":  len(hist_keys),
        "page_cache_entries": len(page_keys),
        "ttl_sec":          _SESSION_TTL_SEC,
        "gc_interval_sec":  _GC_INTERVAL_SEC,
        "max_fetch":        MAX_FETCH,
        "default_page_size": DEFAULT_PAGE_SIZE,
    }
