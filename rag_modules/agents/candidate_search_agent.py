"""
candidate_search_agent.py — 招聘者候选人查询 Agent（分页版）

职责：帮助招聘者查询候选人和报名情况。
recruiter 路径：SQL 强制注入 company_id，确保数据隔离（只能查本公司候选人）。
admin 路径：SQL 注入 tenant_id，可查该租户下所有公司的候选人数据。

API 响应格式：
    {
        "intent": "candidate_search",
        "data": {
            "total": 50,
            "candidates": [...],
            "message": "产品经理职位共50条候选人记录"
        },
        "status": "success"
    }
改进：SQL 生成 Prompt 增加 3 个 Few-Shot 示例。
改动：SQL 固定 LIMIT 100，结果写入 session 缓存，返回第 1 页切片。

v2 变更：
    - handle() 新增 tenant_id 参数（admin 路径使用）
    - 新增 _SQL_SYSTEM_ADMIN_TEMPLATE（tenant_id 版 Prompt）
    - _validate_company_id() 扩展为 _validate_scope_filter()，同时支持 company_id 和 tenant_id
    - handle() 入口根据参数选择对应 Prompt 模板
    - 新增 count-only 模式（只查总数时跳过 list_sql）
"""

import json
import logging
import re
from string import Template
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from db.repositories.job_repo import JobRepo
from controller.session_manager import (
    save_page_cache, get_page,
    MAX_FETCH, DEFAULT_PAGE_SIZE,
)

logger = logging.getLogger(__name__)

RESULT_TYPE = "candidates"

# ─────────────────────────────────────────────
# Prompt — recruiter 版（注入 company_id）
#
# 使用 string.Template（$company_id）而非 str.format()，
# 避免 prompt 中 JSON 示例的花括号 {} 与 .format() 占位符冲突。
# ─────────────────────────────────────────────

_SQL_SYSTEM_RECRUITER_TEMPLATE = Template("""你是一个招聘数据库查询助手，正在帮助招聘者查询自己公司的候选人和报名情况。
根据用户的自然语言需求，同时生成两条 SQL 语句。

涉及的表：
    employees_apply（报名表，别名 ea）
    job（职位表，别名 j）
    company（企业表，别名 c）

关联关系：
    ea.job_id = j.id
    ea.company_id = c.id

可用查询条件：
    职位名称：j.name LIKE '%xxx%' 或 ea.job_name LIKE '%xxx%'
    审核状态：ea.status（1审核中 2未通过 3在职 4已离职 5报名取消）
    平台审核：ea.audit_type（1待审核 2录用 3不适合）
    期望薪资：ea.expected_salary
    报名方式：ea.emp_way（0自主 1代替）

重要限制（数据隔离）：
    所有查询必须加上 ea.company_id = $company_id 条件。

多条件规则：
    - 用户提到多个条件时，所有条件用 AND 连接，不要只取其中一个

固定规则：
    1. WHERE 必须包含 ea.company_id = $company_id AND ea.status != 5
    2. list_sql 加 ORDER BY ea.create_time DESC 和 LIMIT 100（固定，不可更改）
    3. count_sql 不加 LIMIT

list_sql 固定 SELECT 字段：
    ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id,
    COALESCE(j.name, ea.job_name) as job_name, ea.company_id,
    COALESCE(c.name, ea.work_company_name) as company_name,
    ea.expected_salary, ea.status, ea.audit_type, ea.emp_way,
    ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason

只输出 JSON，格式：{"count_sql": "...", "list_sql": "...", "message": "..."}

═══════════════════════════════════════════════
Few-Shot 示例（company_id 均用 $company_id）
═══════════════════════════════════════════════

【示例1 — 查询某职位全部候选人】
用户输入: 产品经理职位有多少人报名
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%产品经理%' OR ea.job_name LIKE '%产品经理%')",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%产品经理%' OR ea.job_name LIKE '%产品经理%') ORDER BY ea.create_time DESC LIMIT 100",
    "message": "查询产品经理职位的报名情况"
}

【示例2 — 多条件：职位 + 审核状态】
用户输入: 前端工程师职位待审核的候选人
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%前端%' OR ea.job_name LIKE '%前端%') AND ea.audit_type = 1",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%前端%' OR ea.job_name LIKE '%前端%') AND ea.audit_type = 1 ORDER BY ea.create_time DESC LIMIT 100",
    "message": "查询前端工程师职位待审核的候选人"
}

【示例3 — 查询所有候选人（无额外条件）】
用户输入: 看看公司所有的报名情况
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 ORDER BY ea.create_time DESC LIMIT 100",
    "message": "查询公司所有报名记录"
}
═══════════════════════════════════════════════""")


# ─────────────────────────────────────────────
# Prompt — admin 版（注入 tenant_id）
# ─────────────────────────────────────────────

_SQL_SYSTEM_ADMIN_TEMPLATE = Template("""你是一个招聘数据库查询助手，正在帮助平台管理员查询该租户下所有公司的候选人和报名情况。
根据用户的自然语言需求，同时生成两条 SQL 语句。

涉及的表：
    employees_apply（报名表，别名 ea）
    job（职位表，别名 j）
    company（企业表，别名 c）

关联关系：
    ea.job_id = j.id
    ea.company_id = c.id

可用查询条件：
    职位名称：j.name LIKE '%xxx%' 或 ea.job_name LIKE '%xxx%'
    企业名称：c.name LIKE '%xxx%' 或 ea.work_company_name LIKE '%xxx%'
    审核状态：ea.status（1审核中 2未通过 3在职 4已离职 5报名取消）
    平台审核：ea.audit_type（1待审核 2录用 3不适合）
    期望薪资：ea.expected_salary
    报名方式：ea.emp_way（0自主 1代替）

重要限制（数据隔离）：
    所有查询必须加上 c.tenant_id = $tenant_id 条件（通过 company 表关联过滤）。

多条件规则：
    - 用户提到多个条件时，所有条件用 AND 连接，不要只取其中一个

固定规则：
    1. WHERE 必须包含 c.tenant_id = $tenant_id AND ea.status != 5
    2. list_sql 加 ORDER BY ea.create_time DESC 和 LIMIT 100（固定，不可更改）
    3. count_sql 不加 LIMIT

list_sql 固定 SELECT 字段：
    ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id,
    COALESCE(j.name, ea.job_name) as job_name, ea.company_id,
    COALESCE(c.name, ea.work_company_name) as company_name,
    ea.expected_salary, ea.status, ea.audit_type, ea.emp_way,
    ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason

只输出 JSON，格式：{"count_sql": "...", "list_sql": "...", "message": "..."}

═══════════════════════════════════════════════
Few-Shot 示例（tenant_id 均用 $tenant_id）
═══════════════════════════════════════════════

【示例1 — 查询某职位全部候选人】
用户输入: 产品经理职位有多少人报名
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE c.tenant_id = $tenant_id AND ea.status != 5 AND (j.name LIKE '%产品经理%' OR ea.job_name LIKE '%产品经理%')",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE c.tenant_id = $tenant_id AND ea.status != 5 AND (j.name LIKE '%产品经理%' OR ea.job_name LIKE '%产品经理%') ORDER BY ea.create_time DESC LIMIT 100",
    "message": "查询租户下产品经理职位的报名情况"
}

【示例2 — 查询所有候选人】
用户输入: 查看租户下所有报名情况
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE c.tenant_id = $tenant_id AND ea.status != 5",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE c.tenant_id = $tenant_id AND ea.status != 5 ORDER BY ea.create_time DESC LIMIT 100",
    "message": "查询租户下所有报名记录"
}
═══════════════════════════════════════════════""")


# ─────────────────────────────────────────────
# count-only 模式
# ─────────────────────────────────────────────

_COUNT_ONLY_KEYWORDS: list[str] = [
    "有多少", "总数", "数量", "几个", "几家", "几条",
    "多少人", "多少个", "多少条", "统计一下", "共有多少",
]

_LIST_SIGNALS: list[str] = [
    "列出", "列表", "展示", "显示", "看看", "有哪些", "详情",
]


def _is_count_only(query: str) -> bool:
    """
    判断用户是否只需要总数。
    排除同时含有列表类词语的情况（如"有多少候选人，列出来"）。
    """
    has_count = any(kw in query for kw in _COUNT_ONLY_KEYWORDS)
    has_list  = any(kw in query for kw in _LIST_SIGNALS)
    return has_count and not has_list


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _llm_call(llm: ChatOpenAI, system: str, user_content: str,
              history: Optional[list[dict]] = None) -> str:
    """
    直接用 Message 对象构造消息列表，不经过 ChatPromptTemplate。
    避免 system prompt 中的 JSON 花括号被 LangChain 误识别为模板变量。
    """
    history  = history or []
    messages = [SystemMessage(content=system)]
    for turn in history:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    messages.append(HumanMessage(content=user_content))
    return (llm | StrOutputParser()).invoke(messages).strip()


def _parse_json_safe(text: str) -> Optional[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.splitlines()
        inner   = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[candidate_search_agent] JSON 解析失败: {e}")
        return None


def _validate_scope_filter(sql: str, company_id: Optional[int], tenant_id: Optional[int]) -> bool:
    """
    第三道防线：校验 SQL 包含对应的范围过滤条件。
    recruiter 模式校验 company_id，admin 模式校验 tenant_id。
    扩展自原 _validate_company_id()，同时支持两种过滤方式。
    """
    if company_id is not None:
        return bool(
            re.compile(rf"(ea\.)?company_id\s*=\s*{company_id}", re.IGNORECASE).search(sql)
        )
    if tenant_id is not None:
        return bool(
            re.compile(rf"(ea\.|j\.|c\.)?tenant_id\s*=\s*{tenant_id}", re.IGNORECASE).search(sql)
        )
    return False


def _error_response(message):
    return {"intent": "candidate_search", "data": {"message": message}, "status": "error"}


def _build_response(page_data, total_db, message):
    return {
        "intent": "candidate_search",
        "data": {
            "message":     message,
            "candidates":  page_data["items"],
            "pagination": {
                "page":        page_data["page"],
                "page_size":   page_data["page_size"],
                "total_pages": page_data["total_pages"],
                "fetched":     page_data["fetched"],
                "total_db":    total_db,
            },
        },
        "status": "success",
    }


def handle(
    query:      str,
    session_id: str,
    company_id: Optional[int] = None,
    tenant_id:  Optional[int] = None,
    history:    Optional[list[dict]] = None,
    llm = None,
    page_size:  int = DEFAULT_PAGE_SIZE,
) -> dict:
    """
    招聘者候选人查询 Agent 主入口。

    Args:
        query:      用户输入
        company_id: 招聘者关联的企业 ID（recruiter 路径，由控制器层强制注入）
        tenant_id:  租户 ID（admin 路径，由控制器层强制注入）
                    company_id 和 tenant_id 必须有且仅有一个不为 None
        history:    多轮对话历史（最多 5 轮）
        llm:        ChatOpenAI 实例

    Returns:
        统一响应字典
    """
    history = history or []

    if llm is None:
        logger.error("[candidate_search_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    if company_id is None and tenant_id is None:
        logger.error("[candidate_search_agent] company_id 和 tenant_id 均为 None，缺少范围过滤参数")
        return _error_response("缺少必要的权限参数，请联系管理员")

    # 根据参数选择对应模板（Template.substitute 不受 JSON 花括号干扰）
    if company_id is not None:
        system = _SQL_SYSTEM_RECRUITER_TEMPLATE.substitute(company_id=company_id)
    else:
        system = _SQL_SYSTEM_ADMIN_TEMPLATE.substitute(tenant_id=tenant_id)

    raw    = _llm_call(llm, system, query, history)
    parsed = _parse_json_safe(raw)

    if parsed is None or "count_sql" not in parsed or "list_sql" not in parsed:
        logger.error(f"[candidate_search_agent] LLM 返回格式异常: {raw[:200]}")
        return _error_response("意图解析失败，请描述您想查询哪个职位的候选人")

    count_sql = parsed["count_sql"]
    list_sql  = parsed["list_sql"]
    llm_msg   = parsed.get("message", "查询候选人")

    # 第三道防线：SQL 安全校验
    for sql_name, sql in [("count_sql", count_sql), ("list_sql", list_sql)]:
        if not _validate_scope_filter(sql, company_id, tenant_id):
            logger.error(
                f"[candidate_search_agent] 安全校验失败：{sql_name} "
                f"缺少 company_id={company_id} 或 tenant_id={tenant_id}"
            )
            return _error_response("查询生成异常，请重试或联系管理员")

    logger.info(f"[candidate_search_agent] count_sql: {count_sql}")
    logger.info(f"[candidate_search_agent] list_sql:  {list_sql}")

    repo     = JobRepo()

    # count-only 模式：只查总数，跳过 list_sql
    count_only = _is_count_only(query)
    total_db   = repo.execute_count_query(count_sql)

    if count_only:
        if total_db < 0:
            total_db = 0
        return {
            "intent": "candidate_search",
            "data": {
                "message":    f"{llm_msg}，共 {total_db} 条记录",
                "candidates": [],
                "pagination": None,   # 无分页数据，前端不渲染翻页组件
            },
            "status": "success",
        }

    candidates = repo.execute_apply_query(list_sql)

    if total_db < 0:
        total_db = len(candidates)

    if not candidates:
        return {
            "intent": "candidate_search",
            "data": {
                "message": f"{llm_msg}，暂无符合条件的候选人",
                "candidates": [],
                "pagination": {
                    "page": 1, "page_size": page_size,
                    "total_pages": 0, "fetched": 0, "total_db": 0,
                },
            },
            "status": "success",
        }

    save_page_cache(
        session_id  = session_id,
        result_type = RESULT_TYPE,
        items       = candidates,
        total_db    = total_db,
        query       = query,
    )

    page_data = get_page(session_id, RESULT_TYPE, page=1, page_size=page_size)
    fetched   = len(candidates)
    hint = f"（数据库共 {total_db} 条，已为您加载前 {fetched} 条）" if total_db > fetched else ""
    message = f"{llm_msg}，共 {fetched} 条记录{hint}"

    return _build_response(page_data, total_db, message)
