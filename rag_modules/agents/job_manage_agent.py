"""
job_manage_agent.py — 招聘者职位管理 Agent（分页版）

职责：帮助招聘者查看/管理自己公司发布的职位。
recruiter 路径：SQL 强制注入 company_id，确保数据隔离（只能查本公司数据）。
admin 路径：SQL 注入 tenant_id，可查该租户下所有公司的职位数据。

API 响应格式：
    {
        "intent": "job_manage",
        "data": {
            "total": 5,
            "jobs": [...],
            "message": "您公司当前有5个在招职位"
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

RESULT_TYPE = "jobs"

# ─────────────────────────────────────────────
# Prompt — recruiter 版（注入 company_id）
# ─────────────────────────────────────────────

_SQL_SYSTEM_RECRUITER_TEMPLATE = Template("""你是一个招聘数据库查询助手，正在帮助招聘者查看自己公司发布的职位。
根据用户的自然语言需求，同时生成两条查询 job 表的 SQL 语句。

表名：job
可用查询字段：
    name         VARCHAR  职位名称，用 LIKE '%xxx%' 模糊匹配
    status       TINYINT  职位状态：0未审核 1已发布 2不通过 3停止发布
    job_type     TINYINT  职位类型：0全职 1就业 2实习 3临时工
    audit_status TINYINT  审核状态：0未审核 1通过 2不通过
    work_city    VARCHAR  工作城市
    salary_min   INT      最低薪资
    salary_max   INT      最高薪资
    create_time  DATETIME 创建时间
    deploy_time  DATETIME 发布时间

重要限制（数据隔离）：
    所有查询必须加上 company_id = $company_id 条件。

固定规则：
    1. WHERE 必须包含 company_id = $company_id AND is_delete = 0
    2. 查询"在招/发布中"的职位时加 status=1 AND audit_status=1
    3. 用户未指定状态时默认查询全部未删除职位（不加 status 过滤）
    4. list_sql 加 LIMIT 100（固定，不可更改）和 ORDER BY deploy_time DESC
    5. list_sql SELECT 字段固定为：
       id, name, company_name, company_logo, salary, salary_min, salary_max,
       job_exp, education, job_type, job_duty, work_city,
       contact_name, contact_phone, welfare, status, audit_status
    6. count_sql 固定为：SELECT COUNT(*) AS total FROM job WHERE ...

只输出 JSON，格式：{"count_sql": "...", "list_sql": "...", "message": "..."}

═══════════════════════════════════════════════
Few-Shot 示例（company_id 均用 $company_id）
═══════════════════════════════════════════════

【示例1 — 查询在招职位】
用户输入: 我们公司现在有哪些在招职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 1 AND audit_status = 1",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 1 AND audit_status = 1 ORDER BY deploy_time DESC LIMIT 100",
    "message": "查询公司当前在招职位"
}

【示例2 — 多条件：职位类型 + 城市】
用户输入: 公司在深圳发布的全职职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0 AND work_city LIKE '%深圳%' AND job_type = 0",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 AND work_city LIKE '%深圳%' AND job_type = 0 ORDER BY deploy_time DESC LIMIT 100",
    "message": "查询公司在深圳发布的全职职位"
}

【示例3 — 查询全部职位（不限状态）】
用户输入: 公司所有职位列表
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 ORDER BY deploy_time DESC LIMIT 100",
    "message": "查询公司全部职位"
}

【示例4 — 查询停止发布的职位】
用户输入: 查一下已下线的职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 3",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 3 ORDER BY deploy_time DESC LIMIT 100",
    "message": "查询公司已停止发布的职位"
}
═══════════════════════════════════════════════""")


# ─────────────────────────────────────────────
# Prompt — admin 版（注入 tenant_id）
# ─────────────────────────────────────────────

_SQL_SYSTEM_ADMIN_TEMPLATE = Template("""你是一个招聘数据库查询助手，正在帮助平台管理员查看该租户下所有公司发布的职位。
根据用户的自然语言需求，同时生成两条查询 job 表的 SQL 语句。

表名：job
可用查询字段：
    name         VARCHAR  职位名称，用 LIKE '%xxx%' 模糊匹配
    company_name VARCHAR  企业名称，用 LIKE '%xxx%' 模糊匹配
    status       TINYINT  职位状态：0未审核 1已发布 2不通过 3停止发布
    job_type     TINYINT  职位类型：0全职 1就业 2实习 3临时工
    audit_status TINYINT  审核状态：0未审核 1通过 2不通过
    work_city    VARCHAR  工作城市
    salary_min   INT      最低薪资
    salary_max   INT      最高薪资
    create_time  DATETIME 创建时间
    deploy_time  DATETIME 发布时间

重要限制（数据隔离）：
    所有查询必须加上 tenant_id = $tenant_id 条件。

固定规则：
    1. WHERE 必须包含 tenant_id = $tenant_id AND is_delete = 0
    2. 查询"在招/发布中"的职位时加 status=1 AND audit_status=1
    3. 用户未指定状态时默认查询全部未删除职位（不加 status 过滤）
    4. list_sql 加 LIMIT 100（固定，不可更改）和 ORDER BY deploy_time DESC
    5. list_sql SELECT 字段固定为：
       id, name, company_name, company_logo, salary, salary_min, salary_max,
       job_exp, education, job_type, job_duty, work_city,
       contact_name, contact_phone, welfare, status, audit_status
    6. count_sql 固定为：SELECT COUNT(*) AS total FROM job WHERE ...

只输出 JSON，格式：{"count_sql": "...", "list_sql": "...", "message": "..."}

═══════════════════════════════════════════════
Few-Shot 示例（tenant_id 均用 $tenant_id）
═══════════════════════════════════════════════

【示例1 — 查询所有在招职位】
用户输入: 租户下有哪些在招职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE tenant_id = $tenant_id AND is_delete = 0 AND status = 1 AND audit_status = 1",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE tenant_id = $tenant_id AND is_delete = 0 AND status = 1 AND audit_status = 1 ORDER BY deploy_time DESC LIMIT 100",
    "message": "查询租户下所有在招职位"
}

【示例2 — 指定企业名称查询】
用户输入: XX科技公司发布的职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE tenant_id = $tenant_id AND is_delete = 0 AND company_name LIKE '%XX科技%'",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE tenant_id = $tenant_id AND is_delete = 0 AND company_name LIKE '%XX科技%' ORDER BY deploy_time DESC LIMIT 100",
    "message": "查询XX科技公司发布的职位"
}
═══════════════════════════════════════════════""")


# ─────────────────────────────────────────────
# count-only 模式（只查总数，跳过 list_sql）
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
    排除同时含有列表类词语的情况（如"有多少职位，列出来"）。
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
        logger.error(f"[job_manage_agent] JSON 解析失败: {e} | 原文: {text[:200]}")
        return None


def _validate_scope_filter(sql: str, company_id: Optional[int], tenant_id: Optional[int]) -> bool:
    """
    第三道防线：校验 SQL 包含对应的范围过滤条件。
    recruiter 模式校验 company_id，admin 模式校验 tenant_id。
    扩展自原 _validate_company_id()，同时支持两种过滤方式。
    """
    if company_id is not None:
        return bool(
            re.compile(rf"company_id\s*=\s*{company_id}", re.IGNORECASE).search(sql)
        )
    if tenant_id is not None:
        return bool(
            re.compile(rf"(j\.|c\.)?tenant_id\s*=\s*{tenant_id}", re.IGNORECASE).search(sql)
        )
    return False


def _error_response(message: str) -> dict:
    return {"intent": "job_manage", "data": {"message": message}, "status": "error"}


def _build_response(page_data, total_db, message):
    return {
        "intent": "job_manage",
        "data": {
            "message": message,
            "jobs":    page_data["items"],
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

# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

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
    招聘者职位管理 Agent 主入口。

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
        logger.error("[job_manage_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    if company_id is None and tenant_id is None:
        logger.error("[job_manage_agent] company_id 和 tenant_id 均为 None，缺少范围过滤参数")
        return _error_response("缺少必要的权限参数，请联系管理员")

    # 根据参数选择对应模板
    if company_id is not None:
        system = _SQL_SYSTEM_RECRUITER_TEMPLATE.substitute(company_id=company_id)
    else:
        system = _SQL_SYSTEM_ADMIN_TEMPLATE.substitute(tenant_id=tenant_id)

    raw    = _llm_call(llm, system, query, history)
    parsed = _parse_json_safe(raw)

    if parsed is None or "count_sql" not in parsed or "list_sql" not in parsed:
        logger.error(f"[job_manage_agent] LLM 返回格式异常: {raw[:200]}")
        return _error_response("查询解析失败，请重新描述您的需求")

    count_sql = parsed["count_sql"]
    list_sql  = parsed["list_sql"]
    llm_msg   = parsed.get("message", "查询公司职位")

    # 第三道防线：SQL 安全校验
    for sql_name, sql in [("count_sql", count_sql), ("list_sql", list_sql)]:
        if not _validate_scope_filter(sql, company_id, tenant_id):
            logger.error(
                f"[job_manage_agent] 安全校验失败：{sql_name} "
                f"缺少 company_id={company_id} 或 tenant_id={tenant_id}"
            )
            return _error_response("查询生成异常，请重试或联系管理员")

    repo     = JobRepo()

    # count-only 模式：只查总数，跳过 list_sql
    count_only = _is_count_only(query)
    total_db = repo.execute_count_query(count_sql)

    if count_only:
        if total_db < 0:
            total_db = 0
        return {
            "intent": "job_manage",
            "data": {
                "message":    f"{llm_msg}，共 {total_db} 个职位",
                "jobs":       [],
                "pagination": None,   # 无分页数据，前端不渲染翻页组件
            },
            "status": "success",
        }

    jobs = repo.execute_job_query(list_sql)

    if total_db < 0:
        total_db = len(jobs)

    if not jobs:
        return {
            "intent": "job_manage",
            "data": {
                "message": f"{llm_msg}，暂无符合条件的职位",
                "jobs":    [],
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
        items       = jobs,
        total_db    = total_db,
        query       = query,
    )

    page_data = get_page(session_id, RESULT_TYPE, page=1, page_size=page_size)
    fetched   = len(jobs)
    hint = f"（数据库共 {total_db} 个，已为您加载前 {fetched} 个）" if total_db > fetched else ""
    message = f"{llm_msg}，共 {fetched} 个职位{hint}"

    return _build_response(page_data, total_db, message)
