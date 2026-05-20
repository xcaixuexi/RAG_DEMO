from rag_modules.supervisor import Supervisor
from rag_modules.agents import (
    resume_agent,
    match_agent,
    knowledge_agent,
    chitchat_agent,
    unknown_agent,
)


class ChatController:
    """
    对话控制器 - 协调 Supervisor 路由与各 Agent 执行。

    路由流程（两级漏斗）：
        用户输入
        [Level-1] 规则路由 (RuleRouter)
            命中 直接调用 Agent（0 次 LLM）
        未命中
        [Level-2] LLM 路由 (query_rewrite → query_router)
        调用对应 Agent
    """

    def __init__(self):
        self.supervisor = Supervisor(
            temperature=0.0,
            rule_confidence_threshold=1,   # 置信度阈值，可按需调高至 2
            enable_rule_router=True,        # 关闭可退回纯 LLM 路由，便于 A/B 对比
        )
        self.agent_map = {
            "resume_parse": resume_agent.handle,
            "job_match":    match_agent.handle,
            "knowledge":    knowledge_agent.handle,
            "chitchat":     chitchat_agent.handle,
            "unknown":      unknown_agent.handle,
        }

    def process_message(self, user_input: str) -> str:
        """
        处理用户消息，返回 Agent 响应。

        Args:
            user_input: 用户原始输入

        Returns:
            Agent 返回的字符串响应
        """
        # Supervisor.route() 统一处理两级路由，返回 (处理后查询, 意图)
        processed_query, intent = self.supervisor.route(user_input)

        handler = self.agent_map.get(intent, unknown_agent.handle)
        return handler(processed_query)

    def get_routing_stats(self) -> dict:
        """
        获取路由命中率统计（供监控 / 日志 / 管理面板使用）。

        Returns:
            {
                "total": 总请求数,
                "rule_hit": 规则路由命中数,
                "llm_hit": LLM 路由命中数,
                "rule_hit_rate": 0.xx,
                "llm_hit_rate": 0.xx
            }
        """
        return self.supervisor.get_stats()