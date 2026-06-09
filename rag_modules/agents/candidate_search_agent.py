"""
candidate_search_agent.py — 招聘者候选人查询 Agent（Few-Shot 版）

职责：帮助招聘者查询自己公司职位的候选人和报名情况。
仅服务招聘者（recruiter），SQL 强制注入 company_id 过滤实现数据隔离。

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
#
# 使用 string.Template（$company_id）而非 str.format()，
# 避免 prompt 中 JSON 示例的花括号 {} 与 .format() 占位符冲突。
# ─────────────────────────────────────────────

_SQL_SYSTEM_TEMPLATE = Template("""你是一个招聘数据库查询助手，正在帮助招聘者查询自己公司的候选人和报名情况。
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
    2. list_sql 加 ORDER BY ea.create_time DESC 和 LIMIT 50
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
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%产品经理%' OR ea.job_name LIKE '%产品经理%') ORDER BY ea.create_time DESC LIMIT 50",
    "message": "查询产品经理职位的报名情况"
}

【示例2 — 多条件：职位 + 审核状态】
用户输入: 前端工程师职位待审核的候选人
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%前端%' OR ea.job_name LIKE '%前端%') AND ea.audit_type = 1",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 AND (j.name LIKE '%前端%' OR ea.job_name LIKE '%前端%') AND ea.audit_type = 1 ORDER BY ea.create_time DESC LIMIT 50",
    "message": "查询前端工程师职位待审核的候选人"
}

【示例3 — 查询所有候选人（无额外条件）】
用户输入: 看看公司所有的报名情况
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5",
    "list_sql": "SELECT ea.id as apply_id, ea.user_id, ea.resume_id, ea.job_id, COALESCE(j.name, ea.job_name) as job_name, ea.company_id, COALESCE(c.name, ea.work_company_name) as company_name, ea.expected_salary, ea.status, ea.audit_type, ea.emp_way, ea.create_time, ea.audit_time, ea.cancel_time, ea.remark, ea.reason FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ea.company_id = $company_id AND ea.status != 5 ORDER BY ea.create_time DESC LIMIT 50",
    "message": "查询公司所有报名记录"
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
        logger.error(f"[candidate_search_agent] JSON 解析失败: {e} | 原文: {text[:200]}")
        return None


def _validate_company_id_in_sql(sql: str, company_id: int) -> bool:
    """第三道防线：验证 SQL 包含 ea.company_id 或 company_id 过滤"""
    pattern = re.compile(rf"(ea\.)?company_id\s*=\s*{company_id}", re.IGNORECASE)
    return bool(pattern.search(sql))


def _error_response(message: str) -> dict:
    return {"intent": "candidate_search", "data": {"message": message}, "status": "error"}


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
    招聘者候选人查询 Agent 主入口。

    Args:
        query:      用户输入
        company_id: 当前招聘者关联的企业 ID（由控制器层强制注入）
        history:    多轮对话历史（最多 5 轮）
        llm:        ChatOpenAI 实例

    Returns:
        统一响应字典
    """
    history = history or []

    if llm is None:
        logger.error("[candidate_search_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    # 动态渲染 System Prompt，注入 company_id（Template.substitute 不受 JSON 花括号干扰）
    system = _SQL_SYSTEM_TEMPLATE.substitute(company_id=company_id)
    raw    = _llm_call(llm, system, query, history)
    parsed = _parse_json_safe(raw)

    if parsed is None or "count_sql" not in parsed or "list_sql" not in parsed:
        logger.error(f"[candidate_search_agent] LLM 返回格式异常: {raw[:200]}")
        return _error_response("意图解析失败，请描述您想查询哪个职位的候选人")

    count_sql = parsed["count_sql"]
    list_sql  = parsed["list_sql"]
    message   = parsed.get("message", "查询候选人")

    # 第三道防线：SQL 安全校验
    for sql_name, sql in [("count_sql", count_sql), ("list_sql", list_sql)]:
        if not _validate_company_id_in_sql(sql, company_id):
            logger.error(
                f"[candidate_search_agent] 安全校验失败：{sql_name} 缺少 company_id={company_id} "
                f"| SQL: {sql[:100]}"
            )
            return _error_response("查询生成异常，请重试或联系管理员")

    logger.info(f"[candidate_search_agent] count_sql: {count_sql}")
    logger.info(f"[candidate_search_agent] list_sql:  {list_sql}")

    repo       = JobRepo()
    total      = repo.execute_count_query(count_sql)
    candidates = repo.execute_apply_query(list_sql)

    if total < 0:
        total = len(candidates)

    return {
        "intent": "candidate_search",
        "data": {
            "total":      total,
            "candidates": candidates,
            "message":    f"{message}，共 {total} 条记录",
        },
        "status": "success",
    }
