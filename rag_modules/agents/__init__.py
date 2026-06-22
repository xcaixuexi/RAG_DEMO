"""
rag_modules/agents/__init__.py

统一导出所有 Agent 模块，便于控制器层 import。

v2 变更：
    - 新增 platform_stats_agent 的导入与计时注册

v3 变更（方案一）：
    - 文件名保持不变（job_search_agent.py / job_manage_agent.py / candidate_search_agent.py），
      仅在意图层面合并/改名：
        · job_search_agent  → 对应新意图 job_query 的 jobseeker 分支
        · job_manage_agent  → 对应新意图 job_query 的 recruiter/admin 分支
        · candidate_search_agent → 对应新意图 candidate_query（文件名不变，逻辑不变）
      计时标签沿用原模块名，方便日志按文件定位，不强行对齐新意图名。
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
# v3 说明：job_search_agent / job_manage_agent 现共同服务于 job_query 意图，
#          candidate_search_agent 服务于 candidate_query 意图，
#          文件名与计时标签保持不变（按文件定位日志更直观）。
_agents = [
    ("resume_agent",           resume_agent),
    ("job_search_agent",       job_search_agent),        # job_query → jobseeker 路径
    ("job_manage_agent",       job_manage_agent),        # job_query → recruiter/admin 路径
    ("candidate_search_agent", candidate_search_agent),  # candidate_query
    ("knowledge_agent",        knowledge_agent),
    ("chitchat_agent",         chitchat_agent),
    ("unknown_agent",          unknown_agent),
    ("platform_stats_agent",   platform_stats_agent),
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
    "platform_stats_agent",
]
