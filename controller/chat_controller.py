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
    第三道：Agent 层 — job_manage/candidate_search 的 SQL 强制携带 company_id 或 tenant_id

主要改进：
    1. 使用 Supervisor.get_instance() 获取单例，不再每次请求新建
    2. route() 调用时传入上一轮 intent/query，支持上下文感知路由
    3. session 中记录 last_intent/last_query，供下次请求读取

v2 变更：
    - 新增 admin 角色，白名单加入 platform_stats
    - __init__() 阶段对 admin 角色执行 tenant_id 鉴权
    - 新增 _get_tenant_id()，根据 user_id 查询 platform 类型企业取 tenant_id
    - _call_job_manage() / _call_candidate_search() 新增 admin 分支，注入 tenant_id
    - 新增 _call_job_manage_admin() / _call_candidate_search_admin()
    - 新增 _call_platform_stats()，调用 platform_stats_agent
    - _PERMISSION_DENIED_MESSAGES 新增 admin 相关条目

v3 变更：
    - admin 权限矩阵改为 None（不做意图拦截，所有意图均可访问）
    - _check_permission() 新方法统一处理权限校验，None 白名单直接放行
    - process_message() 权限校验改为调用 _check_permission()
    - _call_job_search() 新增 admin 分支：透传 tenant_id 给 job_search_agent
    - _call_job_manage() / _call_candidate_search() admin 分支保持不变（已有）
    - 移除 ("admin", "job_search") 的 permission_denied 条目（admin 现可访问 job_search）

v4 变更（方案一：意图按操作类型重构）：
    - _ROLE_INTENT_WHITELIST：job_search/job_manage 合并为 job_query；
                              candidate_search 改名为 candidate_query
    - _PERMISSION_DENIED_MESSAGES：
        · ("jobseeker", "candidate_search") → ("jobseeker", "candidate_query")，内容不变
        · ("jobseeker", "job_manage") 删除（job_manage 已并入 job_query，
          jobseeker 访问 job_query 会被分发到 job_search_agent，不存在越权）
        · ("recruiter", "job_search") 删除（job_search 已并入 job_query，
          recruiter 访问 job_query 会被分发到 job_manage_agent，不存在越权）
    - _agent_map：
        · "job_search"/"job_manage" 两个 key 替换为统一的 "job_query" → _call_job_query
        · "candidate_search" key 改名为 "candidate_query"（方法体沿用 _call_candidate_search，逻辑不变）
    - 新增 _call_job_query()：按角色分发到 job_search_agent（jobseeker）
      或 job_manage_agent（recruiter 注入 company_id / admin 注入 tenant_id）
    - 删除 _call_job_search()、_call_job_manage()、_call_job_manage_admin()，
      三者逻辑完整搬迁进 _call_job_query()，无任何逻辑变更
    - _call_candidate_search() / _call_candidate_search_admin() 方法名保留不变
      （仅 agent_map 中的 key 改名为 candidate_query，避免无意义的大范围改名）
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
    platform_stats_agent,
)
from controller import session_manager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 权限控制常量（完整版，含 admin）
# v4 变更：job_search/job_manage → job_query；candidate_search → candidate_query
# ─────────────────────────────────────────────

# admin 值为 None，表示不做意图拦截——所有意图均可访问。
# admin 可以调用 job_query/candidate_query 时，
# 控制器层在分发方法内注入 tenant_id 实现数据隔离（见各 _call_* 方法）。
_ROLE_INTENT_WHITELIST: dict[str, set | None] = {
    "jobseeker": {
        "resume_parse", "job_query", "knowledge", "chitchat", "unknown",
    },
    "recruiter": {
        "resume_parse", "job_query", "candidate_query", "knowledge", "chitchat", "unknown",
    },
    "admin": None,  # None 表示不做意图拦截，所有意图均可访问（数据隔离由 tenant_id 保证）
}

_PERMISSION_DENIED_MESSAGES: dict = {
    # 求职者触发了招聘者功能
    ("jobseeker", "candidate_query"): (
        "抱歉，候选人查询功能仅限招聘者使用。"
        "如需搜索职位，可以告诉我您的求职需求，例如城市、岗位或薪资要求 😊"
    ),
    # v4 说明：("jobseeker", "job_manage") 条目已删除。
    # job_search 和 job_manage 合并为 job_query 后，jobseeker 访问 job_query
    # 会被 _call_job_query() 分发到 job_search_agent（全平台搜索），不存在越权场景。
    #
    # v3 说明：admin 白名单已改为 None（不拦截），("admin", "job_search") 条目已移除。
    # admin 调用 job_query 时会走 _call_job_query() 的 admin 分支，注入 tenant_id。
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
        - admin 角色在 __init__() 阶段完成 tenant_id 鉴权

    v4 变更：
        - _agent_map 中 job_search/job_manage 合并为 job_query → _call_job_query()
        - candidate_search 改名为 candidate_query（方法体仍是 _call_candidate_search）
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

        # admin 鉴权：在初始化阶段就检查 tenant_id，失败时标记鉴权失败
        # 鉴权结果缓存在 _auth_failed 和 _cached_tenant_id 中
        self._auth_failed:       bool          = False
        self._cached_tenant_id:  Optional[int] = None
        if self.user_role == "admin":
            self._init_admin_auth()

        # ── 获取单例 Supervisor（不重复初始化 LLM）──
        self.supervisor = Supervisor.get_instance(
            rule_confidence_threshold = 1,
            enable_rule_router        = True,
        )

        # v4：job_search / job_manage 合并为统一的 job_query 入口
        #     candidate_search 改名为 candidate_query（方法体沿用 _call_candidate_search）
        self._agent_map = {
            "resume_parse":     self._call_resume,
            "job_query":        self._call_job_query,
            "candidate_query":  self._call_candidate_search,
            "platform_stats":   self._call_platform_stats,
            "knowledge":        self._call_knowledge,
            "chitchat":         self._call_chitchat,
            "unknown":          self._call_unknown,
        }

    # ==================== 权限校验 ====================

    def _check_permission(self, intent: str) -> bool:
        """
        校验当前角色是否有权访问指定意图。
        - whitelist 为 None（admin）→ 直接放行，所有意图均可访问
        - whitelist 为 set      → intent 必须在集合内
        v3 新增，替换原 process_message() 中的内联判断逻辑。
        """
        whitelist = _ROLE_INTENT_WHITELIST.get(self.user_role)
        if whitelist is None:
            return True     # admin：不做意图拦截
        return intent in whitelist

    # ==================== admin 鉴权 ====================

    def _init_admin_auth(self) -> None:
        """
        admin 鉴权链路：
            user_id → 查询 company 表
                WHERE user_id=? AND company_type='platform' AND apply_status=1 AND is_delete=0
                → 取到记录 → 提取 tenant_id → 存入 self._cached_tenant_id
                → 未取到记录 → 设置 self._auth_failed = True
        """
        try:
            from db.mysql_client import MySQLClient
            db = MySQLClient.get_instance()
            with db._session() as session:
                from sqlalchemy import text
                row = session.execute(
                    text(
                        "SELECT tenant_id FROM company "
                        "WHERE user_id = :uid AND company_type = 'platform' "
                        "AND apply_status = 1 AND is_delete = 0 LIMIT 1"
                    ),
                    {"uid": self.user_id},
                ).fetchone()

            if row and row[0] is not None:
                self._cached_tenant_id = int(row[0])
                logger.info(f"[admin鉴权] user_id={self.user_id} 关联 tenant_id={self._cached_tenant_id}")
            else:
                self._auth_failed = True
                logger.warning(f"[admin鉴权] user_id={self.user_id} 未找到 platform 类型企业，鉴权失败")
        except Exception as e:
            self._auth_failed = True
            logger.error(f"[admin鉴权] 数据库查询异常: {e}")

    def _get_tenant_id(self) -> Optional[int]:
        """
        返回 admin 的 tenant_id（已在 __init__ 阶段缓存）。
        非 admin 角色返回 None。
        """
        return self._cached_tenant_id

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

        新增：
        - admin 鉴权失败时在入口直接拦截，不进入路由流程
        - 读取上一轮路由信息并传入 supervisor.route()，解决指代/省略句问题
        """
        # admin 鉴权失败时直接拦截
        if self.user_role == "admin" and self._auth_failed:
            return {
                "intent": "auth_failed",
                "data": {
                    "message": (
                        "您的账号未关联平台企业，无法以管理员身份登录，"
                        "请联系系统管理员。"
                    )
                },
                "status": "error",
            }

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

            # 权限校验（v3：通过 _check_permission() 统一处理，admin 直接放行）
            if not self._check_permission(intent):
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

    # ==================== company_id 获取 ====================

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

    def _call_job_query(self, query: str) -> dict:
        """
        v4 新增：统一职位查询入口，按角色分发：
            jobseeker → job_search_agent（无 company_id / tenant_id 限制，搜索全平台公开职位）
            recruiter → job_manage_agent（注入 company_id，只能查本公司职位）
            admin     → job_manage_agent（注入 tenant_id，可查租户下所有公司职位，
                        覆盖原 platform_stats 中迁出的职位统计场景）

        三段逻辑分别完整搬迁自原 _call_job_search()、_call_job_manage()、
        _call_job_manage_admin()，无任何逻辑变更。
        """
        if self.user_role == "jobseeker":
            # ── 原 _call_job_search() jobseeker 分支逻辑 ──────────
            history  = self._get_history("job_search")
            response = job_search_agent.handle(
                query      = query,
                session_id = self.session_id,
                history    = history,
                llm        = self.supervisor.llm,
            )
            self._append_history("job_search", query, response["data"]["message"])
            return response

        elif self.user_role == "recruiter":
            # ── 原 _call_job_manage() recruiter 分支逻辑 ──────────
            company_id = self._get_company_id()
            if company_id is None:
                return {
                    "intent": "job_query",
                    "data":   {"message": "您的账号暂未关联企业信息，请联系管理员完成企业认证后再试。"},
                    "status": "error",
                }
            history  = self._get_history("job_manage")
            response = job_manage_agent.handle(
                query      = query,
                session_id = self.session_id,
                company_id = company_id,
                history    = history,
                llm        = self.supervisor.llm,
            )
            self._append_history("job_manage", query, response["data"]["message"])
            return response

        else:
            # ── 原 _call_job_manage_admin() 逻辑 ──────────────────
            tenant_id = self._get_tenant_id()
            if tenant_id is None:
                return {
                    "intent": "job_query",
                    "data":   {"message": "管理员账号 tenant_id 获取失败，请联系系统管理员。"},
                    "status": "error",
                }
            history  = self._get_history("job_manage")
            response = job_manage_agent.handle(
                query      = query,
                session_id = self.session_id,
                company_id = None,
                tenant_id  = tenant_id,
                history    = history,
                llm        = self.supervisor.llm,
            )
            self._append_history("job_manage", query, response["data"]["message"])
            return response

    def _call_candidate_search(self, query: str) -> dict:
        """
        候选人查询分发（v4：对应新意图名 candidate_query，方法名保留不变）：
            recruiter → 注入 company_id（只能查本公司候选人）
            admin     → 注入 tenant_id（可查租户下所有候选人，
                        覆盖原 platform_stats 中迁出的报名统计场景）
        """
        if self.user_role == "admin":
            return self._call_candidate_search_admin(query)

        # recruiter 原有逻辑
        company_id = self._get_company_id()
        if company_id is None:
            return {
                "intent": "candidate_query",
                "data":   {"message": "您的账号暂未关联企业信息，请联系管理员完成企业认证后再试。"},
                "status": "error",
            }
        history  = self._get_history("candidate_search")
        response = candidate_search_agent.handle(
            query      = query,
            session_id = self.session_id,
            company_id = company_id,
            history    = history,
            llm        = self.supervisor.llm,
        )
        self._append_history("candidate_search", query, response["data"]["message"])
        return response

    def _call_candidate_search_admin(self, query: str) -> dict:
        """admin 版候选人查询：注入 tenant_id，不限 company_id"""
        tenant_id = self._get_tenant_id()
        if tenant_id is None:
            return {
                "intent": "candidate_query",
                "data":   {"message": "管理员账号 tenant_id 获取失败，请联系系统管理员。"},
                "status": "error",
            }
        history  = self._get_history("candidate_search")
        response = candidate_search_agent.handle(
            query      = query,
            session_id = self.session_id,
            company_id = None,
            tenant_id  = tenant_id,
            history    = history,
            llm        = self.supervisor.llm,
        )
        self._append_history("candidate_search", query, response["data"]["message"])
        return response

    def _call_platform_stats(self, query: str) -> dict:
        """
        平台统计（admin 专属）。
        调用 platform_stats_agent，传入 tenant_id。
        v4：platform_stats_agent 职责已缩减为纯企业统计 + 复杂分析，
            职位/候选人统计由 job_query / candidate_query 的 admin 分支承接。
        """
        tenant_id = self._get_tenant_id()
        if tenant_id is None:
            return {
                "intent": "platform_stats",
                "data":   {"message": "管理员账号 tenant_id 获取失败，请联系系统管理员。"},
                "status": "error",
            }
        response = platform_stats_agent.handle(
            query     = query,
            tenant_id = tenant_id,
            llm       = self.supervisor.llm,
        )
        return response

    def _call_unknown(self, query: str) -> dict:
        return unknown_agent.handle(
            query     = query,
            user_role = self.user_role,
            history   = [],
            llm       = self.supervisor.llm,
        )
