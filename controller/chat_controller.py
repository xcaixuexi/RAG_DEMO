"""
controller/chat_controller.py — 对话控制器（重构版）

职责：
    1. 持有 Supervisor 实例，调用两级路由（规则层 + LLM 层）
    2. 接收并存储 user_role / user_id / session_id，透传给所有 Agent
    3. 历史管理委托给 session_manager，通过 _get_history / _append_history 读写
    4. 提供 process_message（文本对话）和 process_file（文件上传）两个入口
    5. 将 Agent 返回的统一响应字典透传给 View 层

权限控制三道防线：
    第一道：LLM 路由层（supervisor.py）— 感知角色，减少越权意图生成概率
    第二道：控制器层（本文件）— 白名单硬校验，意图不在白名单直接返回 permission_denied
    第三道：Agent 层 — job_manage/candidate_search 的 SQL 强制携带 company_id
主要改进：
    1. 使用 Supervisor.get_instance() 获取单例，不再每次请求新建
    2. route() 调用时传入上一轮 intent/query，支持上下文感知路由
    3. session 中记录 last_intent/last_query，供下次请求读取
"""

import logging
from typing import Optional

from rag_modules.supervisor import Supervisor
from rag_modules.agents import (
    resume_agent,
    job_search_agent,
    job_manage_agent,
    candidate_search_agent,
    knowledge_agent,
    chitchat_agent,
    unknown_agent,
)
from controller import session_manager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 权限控制常量（不变）
# ─────────────────────────────────────────────

_ROLE_INTENT_WHITELIST: dict[str, set[str]] = {
    "jobseeker": {
        "resume_parse", "job_search", "knowledge", "chitchat", "unknown",
    },
    "recruiter": {
        "resume_parse", "job_manage", "candidate_search", "knowledge", "chitchat", "unknown",
    },
}

_PERMISSION_DENIED_MESSAGES: dict = {
    # 求职者触发了招聘者功能
    ("jobseeker", "candidate_search"): (
        "抱歉，候选人查询功能仅限招聘者使用。"
        "如需搜索职位，可以告诉我您的求职需求，例如城市、岗位或薪资要求 😊"
    ),
    ("jobseeker", "job_manage"): (
        "抱歉，职位管理功能仅限招聘者使用。"
        "如需查找工作，可以直接告诉我您想找什么类型的职位。"
    ),
    # 招聘者触发了求职者功能
    ("recruiter", "job_search"): (
        "抱歉，您当前以招聘者身份登录，平台求职功能不对招聘者开放。"
        "如需查看公司职位，请说'查看我们公司的职位'；"
        "如需查看候选人，请告诉我职位名称。"
    ),
    # 通用兜底
    "default": "抱歉，您当前身份无权使用该功能，请确认操作是否正确。",
}


class ChatController:
    """
    对话控制器（单例 Supervisor 版）。

    变化：
        - self.supervisor 通过 Supervisor.get_instance() 获取，进程内共享
        - process_message() 读取并传递 prev_intent/prev_query，实现上下文路由
        - process_message() 路由完成后将本轮 intent/query 写回 session
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
        self._pending_file_path = None

        # ── 获取单例 Supervisor（不重复初始化 LLM）──
        self.supervisor = Supervisor.get_instance(
            rule_confidence_threshold = 1,
            enable_rule_router        = True,
        )

        self._agent_map = {
            "resume_parse":     self._call_resume,
            "job_search":       self._call_job_search,
            "job_manage":       self._call_job_manage,
            "candidate_search": self._call_candidate_search,
            "knowledge":        self._call_knowledge,
            "chitchat":         self._call_chitchat,
            "unknown":          self._call_unknown,
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

    def _get_last_route(self) -> tuple[Optional[str], Optional[str]]:
        """
        读取上一轮路由结果（intent + query），用于上下文感知路由。
        存储在 session 的特殊 key "__route__" 中。
        """
        history = session_manager.get_history(self.session_id, "__route__")
        if not history:
            return None, None
        # 最后两条是 {"role": "user", "content": query} 和 {"role": "assistant", "content": intent}
        if len(history) >= 2:
            return history[-1]["content"], history[-2]["content"]
        return None, None

    def _save_last_route(self, query: str, intent: str) -> None:
        """将本轮路由结果写入 session，供下次请求继承"""
        session_manager.append_history(self.session_id, "__route__", query, intent)

    # ==================== 对外主接口 ====================

    def process_message(self, user_input: str) -> dict:
        """
        处理纯文本对话，返回统一响应字典。
        包含完整的两级路由 + 权限校验流程。
        Args:
            user_input: 用户原始输入

        Returns:
            {"intent": "...", "data": {"message": "..."}, "status": "success"/"error"}
        新增：读取上一轮路由信息并传入 supervisor.route()，解决指代/省略句问题。
        """
        try:
            # 读取上下文
            prev_intent, prev_query = self._get_last_route()

            # 两级路由（传入历史上下文）
            processed_query, intent = self.supervisor.route(
                user_input,
                self.user_role,
                prev_intent = prev_intent,
                prev_query  = prev_query,
            )

            # 保存本轮路由结果
            self._save_last_route(user_input, intent)

            # 权限校验
            allowed = _ROLE_INTENT_WHITELIST.get(self.user_role, set())
            if intent not in allowed:
                msg = (
                    _PERMISSION_DENIED_MESSAGES.get((self.user_role, intent))
                    or _PERMISSION_DENIED_MESSAGES["default"]
                )
                logger.warning(
                    f"[权限拦截] user_role={self.user_role} intent={intent} "
                    f"query='{user_input}'"
                )
                return {
                    "intent": "permission_denied",
                    "data":   {"message": msg},
                    "status": "error",
                }

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
        两种角色均可使用。
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

    # ==================== company_id 获取（招聘者专用）====================

    def _get_company_id(self) -> Optional[int]:
        """
        根据 user_id 查询招聘者关联的 company_id。
        结果缓存在实例变量中，同一请求只查一次数据库。
        查不到时返回 None，调用方负责返回错误提示。
        """
        if hasattr(self, "_cached_company_id"):
            return self._cached_company_id
        from db.mysql_client import MySQLClient
        db = MySQLClient.get_instance()
        company = db.get_company_by_user_id(self.user_id)
        self._cached_company_id = company.id if company else None
        if self._cached_company_id is None:
            logger.warning(f"[权限] user_id={self.user_id} 未关联已审核企业")
        return self._cached_company_id

    # ==================== Agent 调用封装 ====================

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

    def _call_resume(self, query: str) -> dict:
        """
        文本路由命中 resume_parse 时的入口。
        CLI 场景：_pending_file_path 有值时直接解析文件。
        Web 场景：返回引导提示，实际文件解析由前端调用 process_file() 处理。
        """
        if self._pending_file_path:
            file_path = self._pending_file_path.replace('\\', '/')
            return self.process_file(file_path)
        return {
            "intent": "resume_parse",
            "data":   {"message": "请上传您的简历文件（支持 .pdf 和 .docx 格式），我来为您解析。"},
            "status": "success",
        }

    def _call_job_search(self, query: str) -> dict:
        """求职者职位搜索，无需 company_id。"""
        history  = self._get_history("job_search")
        response = job_search_agent.handle(
            query   = query,
            history = history,
            llm     = self.supervisor.llm,
        )
        self._append_history("job_search", query, response["data"]["message"])
        return response

    def _call_job_manage(self, query: str) -> dict:
        """招聘者查看自己公司职位，强制注入 company_id。"""
        company_id = self._get_company_id()
        if company_id is None:
            return {
                "intent": "job_manage",
                "data":   {"message": "您的账号暂未关联企业信息，请联系管理员完成企业认证后再试。"},
                "status": "error",
            }
        history  = self._get_history("job_manage")
        response = job_manage_agent.handle(
            query      = query,
            company_id = company_id,
            history    = history,
            llm        = self.supervisor.llm,
        )
        self._append_history("job_manage", query, response["data"]["message"])
        return response

    def _call_candidate_search(self, query: str) -> dict:
        """招聘者查候选人，强制注入 company_id。"""
        company_id = self._get_company_id()
        if company_id is None:
            return {
                "intent": "candidate_search",
                "data":   {"message": "您的账号暂未关联企业信息，请联系管理员完成企业认证后再试。"},
                "status": "error",
            }
        history  = self._get_history("candidate_search")
        response = candidate_search_agent.handle(
            query      = query,
            company_id = company_id,
            history    = history,
            llm        = self.supervisor.llm,
        )
        self._append_history("candidate_search", query, response["data"]["message"])
        return response

    def _call_unknown(self, query: str) -> dict:
        return unknown_agent.handle(
            query     = query,
            user_role = self.user_role,
            history   = [],
            llm       = self.supervisor.llm,
        )
