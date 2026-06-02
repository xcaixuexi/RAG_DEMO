"""
chat_controller.py — 对话控制器

职责：
    1. 持有 Supervisor 实例，调用两级路由（规则层 + LLM 层）
    2. 接收并存储 user_role / user_id / session_id，透传给所有 Agent
    3. 历史管理委托给 session_manager，通过 _get_history / _append_history 读写
    4. 提供 process_message（文本对话）和 process_file（文件上传）两个入口
    5. 将 Agent 返回的统一响应字典透传给 View 层
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
from controller import session_manager

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        user_role:  str = "jobseeker",
        user_id:    int = 0,
        session_id: str = "",
    ):
        """
        Args:
            user_role:  当前用户角色，透传给所有 Agent。
                        可选值："recruiter" / "jobseeker" / "admin"
            user_id:    当前登录用户的系统 ID，透传给 Agent，供后续业务扩展使用
            session_id: 前端生成的会话 UUID，用于读写 session_manager 中的历史
        """
        self.user_role  = user_role
        self.user_id    = user_id
        self.session_id = session_id
        self._pending_file_path = None   # CLI 场景暂存待解析的文件路径

        self.supervisor = Supervisor(
            temperature=0.0,
            rule_confidence_threshold=1,
            enable_rule_router=True,
        )

        self._agent_map = {
            "resume_parse": self._call_resume,
            "job_match":    self._call_match,
            "knowledge":    self._call_knowledge,
            "chitchat":     self._call_chitchat,
            "unknown":      self._call_unknown,
        }

    # ==================== 历史读写 ====================

    def _get_history(self, agent_type: str) -> list[dict]:
        """从 session_manager 读取指定 agent 的历史"""
        return session_manager.get_history(self.session_id, agent_type)

    def _append_history(self, agent_type: str, query: str, reply: str) -> None:
        """
        向 session_manager 追加一轮对话。
        match 场景只传 message 摘要，不传 jobs/candidates 列表。
        """
        session_manager.append_history(self.session_id, agent_type, query, reply)

    # ==================== 对外主接口 ====================

    def process_message(self, user_input: str) -> dict:
        """
        处理纯文本对话，返回统一响应字典。

        Args:
            user_input: 用户原始输入

        Returns:
            {"intent": "...", "data": {"message": "..."}, "status": "success"/"error"}
        """
        try:
            processed_query, intent = self.supervisor.route(user_input)
            handler  = self._agent_map.get(intent, self._call_unknown)
            response = handler(processed_query)
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            response = {
                "intent": "unknown",
                "data":   {"message": "系统出现异常，请稍后重试。"},
                "status": "error",
            }
        return response

    def process_file(self, file_path: str) -> dict:
        """
        处理文件上传（简历解析），直接调用 resume_agent，跳过路由。
        resume_agent 每次独立分析，不读写历史。

        Args:
            file_path: 上传文件的本地路径（.pdf 或 .docx）

        Returns:
            统一响应字典（resume_agent 内部按 user_role 分支处理，均不写库）
        """
        try:
            response = resume_agent.handle(
                file_path = file_path,
                user_role = self.user_role,
                history   = [],
                llm       = self.supervisor.llm,
                user_id   = self.user_id,
            )
        except Exception as e:
            logger.error(f"文件处理时出错: {e}")
            response = {
                "intent": "resume_parse",
                "data":   {"message": "文件处理失败，请检查文件格式后重试。"},
                "status": "error",
            }
        return response

    def get_routing_stats(self) -> dict:
        """获取路由命中率统计"""
        return self.supervisor.get_stats()

    def clear_history(self, agent_type: str = None) -> None:
        """
        清空历史。
        agent_type 为 None 时清除该 session 所有 agent 的历史。
        """
        if agent_type:
            session_manager.clear_agent(self.session_id, agent_type)
        else:
            session_manager.clear_session(self.session_id)
        logger.info(f"历史已清空: session={self.session_id} agent={agent_type or 'all'}")

    # ==================== 各 Agent 调用封装 ====================

    def _call_chitchat(self, query: str) -> dict:
        """调用 chitchat_agent，传入历史、llm 实例和 user_role。"""
        history  = self._get_history("chitchat")
        response = chitchat_agent.handle(
            query     = query,
            history   = history,
            llm       = self.supervisor.llm,
            user_role = self.user_role,
        )
        self._append_history("chitchat", query, response["data"]["message"])
        return response

    def _call_knowledge(self, query: str) -> dict:
        """调用 knowledge_agent，传入历史和 llm。"""
        history  = self._get_history("knowledge")
        response = knowledge_agent.handle(
            query     = query,
            user_role = self.user_role,
            history   = history,
            llm       = self.supervisor.llm,
        )
        self._append_history("knowledge", query, response["data"]["message"])
        return response

    def _call_match(self, query: str) -> dict:
        """
        调用 match_agent，传入历史和 llm。
        存历史时只存 message 摘要，不存 jobs/candidates 列表。
        """
        history  = self._get_history("match")
        response = match_agent.handle(
            query     = query,
            user_role = self.user_role,
            history   = history,
            llm       = self.supervisor.llm,
        )
        # 只取 message 字段存历史，不存 jobs/candidates
        self._append_history("match", query, response["data"]["message"])
        return response

    def _call_resume(self, query: str) -> dict:
        """
        文本路由命中 resume_parse 时的入口。
        CLI 场景：_pending_file_path 有值时直接解析文件。
        Web 场景：_pending_file_path 为 None，返回引导提示，
                  实际文件解析由前端上传后调用 process_file() 处理。
        """
        if self._pending_file_path:
            file_path = self._pending_file_path.replace('\\', '/')
            return self.process_file(file_path)
        return {
            "intent": "resume_parse",
            "data":   {"message": "请上传您的简历文件（支持 .pdf 和 .docx 格式），我来为您解析。"},
            "status": "success",
        }

    def _call_unknown(self, query: str) -> dict:
        return unknown_agent.handle(
            query     = query,
            user_role = self.user_role,
            history   = [],
            llm       = self.supervisor.llm,
        )
