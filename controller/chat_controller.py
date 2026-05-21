"""
chat_controller.py — 对话控制器

职责：
    1. 持有 Supervisor 实例，调用两级路由（规则层 + LLM 层）
    2. 管理多轮对话历史（仅 chitchat 意图，最多保留 3 轮）
    3. 将 llm 实例和 history 传给 chitchat_agent，其余 Agent 保持原有签名
    4. 将 Agent 返回的统一响应字典透传给 View 层
"""

import logging
from collections import deque

from rag_modules.supervisor import Supervisor
from rag_modules.agents import (
    resume_agent,
    match_agent,
    knowledge_agent,
    chitchat_agent,
    unknown_agent,
)

logger = logging.getLogger(__name__)

# chitchat 保留的最大轮数（1轮 = 用户1条 + 助手1条）
_MAX_HISTORY_TURNS = 3


class ChatController:
    """
    对话控制器 - 协调 Supervisor 路由与各 Agent 执行。

    路由流程（两级漏斗）：
        用户输入
            ↓
        [Level-1] 规则路由 (RuleRouter)
            命中 ──→ 直接调用 Agent（0 次 LLM）
            ↓ 未命中
        [Level-2] LLM 路由 (query_rewrite → query_router)
            ↓
        调用对应 Agent，返回统一响应字典
    """

    def __init__(self):
        self.supervisor = Supervisor(
            temperature=0.0,
            rule_confidence_threshold=1,
            enable_rule_router=True,
        )

        # chitchat 多轮历史：deque 自动滚动，maxlen = 轮数 × 2（user + assistant）
        self._chitchat_history: deque[dict] = deque(maxlen=_MAX_HISTORY_TURNS * 2)

        # 其他 Agent 保持原有 handle(query) -> str 签名，统一包装为响应字典
        self._agent_map = {
            "resume_parse": self._call_resume,
            "job_match":    self._call_match,
            "knowledge":    self._call_knowledge,
            "chitchat":     self._call_chitchat,
            "unknown":      self._call_unknown,
        }

    # ==================== 对外主接口 ====================

    def process_message(self, user_input: str) -> dict:
        """
        处理用户消息，返回统一响应字典。

        Args:
            user_input: 用户原始输入

        Returns:
            {"intent": "...", "data": {"message": "..."}, "status": "success"/"error"}
        """
        try:
            processed_query, intent = self.supervisor.route(user_input)
            handler = self._agent_map.get(intent, self._call_unknown)
            response = handler(processed_query)
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            response = {
                "intent": "unknown",
                "data": {"message": "系统出现异常，请稍后重试。"},
                "status": "error",
            }

        return response

    def get_routing_stats(self) -> dict:
        """
        获取路由命中率统计（供监控 / 日志 / 管理面板使用）。

        Returns:
            {"total": N, "rule_hit": N, "llm_hit": N, "rule_hit_rate": 0.xx, ...}
        """
        return self.supervisor.get_stats()

    def clear_history(self):
        """清空 chitchat 多轮历史（如用户主动开启新话题时调用）"""
        self._chitchat_history.clear()
        logger.info("chitchat 历史已清空")

    # ==================== 各 Agent 调用封装 ====================

    def _call_chitchat(self, query: str) -> dict:
        """
        调用 chitchat_agent，传入历史和 llm 实例。
        将本轮对话追加到历史中（仅 LLM 回复才有意义，模板回复也记录保持连贯）。
        """
        history = list(self._chitchat_history)

        response = chitchat_agent.handle(
            query=query,
            history=history,
            llm=self.supervisor.llm,        # 复用 Supervisor 已初始化的 llm，避免重复创建
        )

        # 追加本轮到历史（deque 满时自动丢弃最早一轮的两条）
        self._chitchat_history.append({"role": "user",      "content": query})
        self._chitchat_history.append({"role": "assistant", "content": response["data"]["message"]})

        return response

    def _call_resume(self, query: str) -> dict:
        result = resume_agent.handle(query)
        return {"intent": "resume_parse", "data": {"message": result}, "status": "success"}

    def _call_match(self, query: str) -> dict:
        result = match_agent.handle(query)
        return {"intent": "job_match", "data": {"message": result}, "status": "success"}

    def _call_knowledge(self, query: str) -> dict:
        result = knowledge_agent.handle(query)
        return {"intent": "knowledge", "data": {"message": result}, "status": "success"}

    def _call_unknown(self, query: str) -> dict:
        result = unknown_agent.handle(query)
        return {"intent": "unknown", "data": {"message": result}, "status": "success"}
