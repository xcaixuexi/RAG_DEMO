"""
rag_modules/agents/__init__.py

统一导出所有 Agent 模块，便于控制器层 import。

v2 变更：
    - 新增 platform_stats_agent 的导入与计时注册
"""

from rag_modules.agents import (
    resume_agent,
    job_search_agent,
    job_manage_agent,
    candidate_search_agent,
    knowledge_agent,
    chitchat_agent,
    unknown_agent,
    platform_stats_agent,
)
from utils.timer import timed

# 统一给所有 Agent 的 handle 函数注册计时
_agents = [
    ("resume_agent",           resume_agent),
    ("job_search_agent",       job_search_agent),
    ("job_manage_agent",       job_manage_agent),
    ("candidate_search_agent", candidate_search_agent),
    ("knowledge_agent",        knowledge_agent),
    ("chitchat_agent",         chitchat_agent),
    ("unknown_agent",          unknown_agent),
    ("platform_stats_agent",   platform_stats_agent),   # 新增
]

for _name, _module in _agents:
    _module.handle = timed(_name)(_module.handle)

__all__ = [
    "resume_agent",
    "job_search_agent",
    "job_manage_agent",
    "candidate_search_agent",
    "knowledge_agent",
    "chitchat_agent",
    "unknown_agent",
    "platform_stats_agent",   # 新增
]
