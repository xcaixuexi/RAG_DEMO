"""
session_manager.py — 服务端内存历史会话存储

key 格式："{session_id}:{agent_type}"
value：deque，自动滚动，满了丢弃最早一轮

各 Agent 配置：
    chitchat  → maxlen=6   (3轮×2)
    knowledge → maxlen=10  (5轮×2)
    match     → maxlen=10  (5轮×2，只存用户提问和简短摘要)

后续迁移数据库时只需替换 get_history / append_history 的存取逻辑，
调用方（ChatController）无需改动。
"""

from collections import deque

# ── 全局存储 ──────────────────────────────────
_store: dict[str, deque] = {}

# ── 各 Agent 的 maxlen 配置 ───────────────────
_MAXLEN: dict[str, int] = {
    "chitchat":  6,    # 3轮×2
    "knowledge": 10,   # 5轮×2
    "match":     10,   # 5轮×2
}


def _key(session_id: str, agent_type: str) -> str:
    return f"{session_id}:{agent_type}"


def get_history(session_id: str, agent_type: str) -> list[dict]:
    """
    读取指定 session + agent 的历史，返回 list[dict]。
    key 不存在时返回空列表。
    """
    return list(_store.get(_key(session_id, agent_type), []))


def append_history(
    session_id: str,
    agent_type: str,
    query:      str,
    reply:      str,
) -> None:
    """
    追加一轮对话到历史。
    deque 不存在时按 agent_type 配置的 maxlen 创建。

    Args:
        session_id: 前端传入的会话 ID
        agent_type: "chitchat" | "knowledge" | "match"
        query:      用户原始输入
        reply:      Agent 回复的 message 字段内容
                    （match 场景只传 message 摘要，不传 jobs/candidates 列表）
    """
    k = _key(session_id, agent_type)
    if k not in _store:
        maxlen = _MAXLEN.get(agent_type, 6)
        _store[k] = deque(maxlen=maxlen)

    _store[k].append({"role": "user",      "content": query})
    _store[k].append({"role": "assistant", "content": reply})


def clear_session(session_id: str) -> None:
    """清除某个 session 下所有 agent 的历史（用户主动退出/重置时调用）"""
    keys_to_del = [k for k in _store if k.startswith(f"{session_id}:")]
    for k in keys_to_del:
        del _store[k]


def clear_agent(session_id: str, agent_type: str) -> None:
    """清除某个 session 下指定 agent 的历史"""
    _store.pop(_key(session_id, agent_type), None)
