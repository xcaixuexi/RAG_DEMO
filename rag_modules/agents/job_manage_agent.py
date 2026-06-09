"""
job_manage_agent.py — 招聘者职位管理 Agent（Few-Shot 版）

职责：帮助招聘者查看/管理自己公司发布的职位。
仅服务招聘者（recruiter），且 SQL 强制注入 company_id 过滤，
确保数据隔离（即便控制器层被绕过，也只能查到本公司数据）。

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

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt（动态注入 company_id + Few-Shot）
# ─────────────────────────────────────────────

_SQL_SYSTEM_TEMPLATE = Template("""你是一个招聘数据库查询助手，正在帮助招聘者查看自己公司发布的职位。
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
    4. list_sql 加 LIMIT 50 和 ORDER BY deploy_time DESC
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
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 1 AND audit_status = 1 ORDER BY deploy_time DESC LIMIT 50",
    "message": "查询公司当前在招职位"
}

【示例2 — 多条件：职位类型 + 城市】
用户输入: 公司在深圳发布的全职职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0 AND work_city LIKE '%深圳%' AND job_type = 0",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 AND work_city LIKE '%深圳%' AND job_type = 0 ORDER BY deploy_time DESC LIMIT 50",
    "message": "查询公司在深圳发布的全职职位"
}

【示例3 — 查询全部职位（不限状态）】
用户输入: 公司所有职位列表
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 ORDER BY deploy_time DESC LIMIT 50",
    "message": "查询公司全部职位"
}

【示例4 — 查询停止发布的职位】
用户输入: 查一下已下线的职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 3",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare, status, audit_status FROM job WHERE company_id = $company_id AND is_delete = 0 AND status = 3 ORDER BY deploy_time DESC LIMIT 50",
    "message": "查询公司已停止发布的职位"
}
═══════════════════════════════════════════════""")


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
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
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


def _validate_company_id_in_sql(sql: str, company_id: int) -> bool:
    """
    第三道防线：验证 LLM 生成的 SQL 确实包含 company_id 过滤。
    LLM 偶尔可能忽略约束，此处做兜底校验。
    """
    pattern = re.compile(rf"company_id\s*=\s*{company_id}", re.IGNORECASE)
    return bool(pattern.search(sql))


def _error_response(message: str) -> dict:
    return {"intent": "job_manage", "data": {"message": message}, "status": "error"}


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query:      str,
    company_id: int,
    history:    Optional[list[dict]] = None,
    llm:        Optional[ChatOpenAI] = None,
) -> dict:
    """
    招聘者职位管理 Agent 主入口。

    Args:
        query:      用户输入
        company_id: 当前招聘者关联的企业 ID（由控制器层强制注入，不可为 None）
        history:    多轮对话历史（最多 5 轮）
        llm:        ChatOpenAI 实例

    Returns:
        统一响应字典
    """
    history = history or []

    if llm is None:
        logger.error("[job_manage_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    system = _SQL_SYSTEM_TEMPLATE.substitute(company_id=company_id)
    raw    = _llm_call(llm, system, query, history)
    parsed = _parse_json_safe(raw)

    if parsed is None or "count_sql" not in parsed or "list_sql" not in parsed:
        logger.error(f"[job_manage_agent] LLM 返回格式异常: {raw[:200]}")
        return _error_response("查询解析失败，请重新描述您的需求")

    count_sql = parsed["count_sql"]
    list_sql  = parsed["list_sql"]
    message   = parsed.get("message", "查询公司职位")

    # 第三道防线：SQL 安全校验
    for sql_name, sql in [("count_sql", count_sql), ("list_sql", list_sql)]:
        if not _validate_company_id_in_sql(sql, company_id):
            logger.error(
                f"[job_manage_agent] 安全校验失败：{sql_name} 缺少 company_id={company_id} "
                f"| SQL: {sql[:100]}"
            )
            return _error_response("查询生成异常，请重试或联系管理员")

    logger.info(f"[job_manage_agent] count_sql: {count_sql}")
    logger.info(f"[job_manage_agent] list_sql:  {list_sql}")

    repo  = JobRepo()
    total = repo.execute_count_query(count_sql)
    jobs  = repo.execute_job_query(list_sql)

    if total < 0:
        total = len(jobs)

    return {
        "intent": "job_manage",
        "data": {
            "total":   total,
            "jobs":    jobs,
            "message": f"{message}，共 {total} 个职位",
        },
        "status": "success",
    }
