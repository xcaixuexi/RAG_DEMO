"""
job_search_agent.py — 求职者职位搜索 Agent

职责：帮助求职者在平台公开职位中搜索符合需求的职位。
仅服务求职者（jobseeker），角色校验由控制器层保证，本 Agent 不做二次校验。

API 响应格式：
    {
        "intent": "job_search",
        "data": {
            "total": 1523,
            "jobs": [...],
            "message": "深圳Python开发职位，共找到1523个职位"
        },
        "status": "success"
    }
"""

import json
import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from db.repositories.job_repo import JobRepo

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

_SQL_SYSTEM = """你是一个招聘数据库查询助手，正在帮助一位求职者搜索平台上的公开职位。
根据用户的自然语言需求，同时生成两条查询 job 表的 SQL 语句。

表名：job
可用查询字段：
    name        VARCHAR  职位名称，用 LIKE '%xxx%' 模糊匹配
    work_city   VARCHAR  工作城市，用 LIKE '%xxx%' 模糊匹配
    salary_min  INT      最低薪资（元/月）
    salary_max  INT      最高薪资（元/月）
    salary      VARCHAR  薪资范围，值域：面议/3k以下/3k-5k/5k-8k/8k-12k/12k-15k/15k-20k/20k以上
    job_exp     VARCHAR  工作经验，值域：不限/应届生/3年及以下/3-5年/5-10年/10年以上
    education   VARCHAR  学历要求，值域：不限/大专/本科/硕士/博士
    job_type    TINYINT  职位类型：0全职 1就业 2实习 3临时工

多条件规则：
    - 用户提到多个条件时，所有条件用 AND 连接，不要只取其中一个
    - 职位类型（全职/实习/临时工等）用 job_type 字段匹配，不要用 name LIKE

固定规则：
    1. WHERE 条件必须包含 status=1 AND is_delete=0 AND audit_status=1
    2. 条件不确定时宁可不加，不要强行猜测
    3. list_sql 必须加 LIMIT 50
    4. list_sql SELECT 字段固定为：
       id, name, company_name, company_logo, salary, salary_min, salary_max,
       job_exp, education, job_type, job_duty, work_city,
       contact_name, contact_phone, welfare
    5. count_sql 固定为：SELECT COUNT(*) AS total FROM job WHERE ...

只输出 JSON，格式：
{{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE ...",
    "list_sql": "SELECT id, name, ... FROM job WHERE ... LIMIT 50",
    "message": "一句话说明搜索意图"
}}"""

_NO_RESULT_SYSTEM = """你是一个友好的招聘助手。
用户搜索职位无结果，请给出简短、友好的建议，引导用户放宽条件重试。
回复控制在 50 字以内，不用列条目，一段话即可。"""


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _llm_call(llm: ChatOpenAI, system: str, user_content: str,
              history: Optional[list[dict]] = None) -> str:
    """
    直接用 Message 对象构造消息列表，不经过 ChatPromptTemplate。
    避免 system prompt 中的 JSON 花括号被 LangChain 误识别为模板变量。
    """
    history = history or []
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
        lines = cleaned.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[job_search_agent] JSON 解析失败: {e} | 原文: {text[:200]}")
        return None


def _error_response(message: str) -> dict:
    return {
        "intent": "job_search",
        "data":   {"message": message},
        "status": "error",
    }


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query:   str,
    history: Optional[list[dict]] = None,
    llm:     Optional[ChatOpenAI] = None,
) -> dict:
    """
    求职者职位搜索 Agent 主入口。

    Args:
        query:   用户输入
        history: 多轮对话历史（最多 5 轮）
        llm:     ChatOpenAI 实例

    Returns:
        统一响应字典
    """
    history = history or []

    if llm is None:
        logger.error("[job_search_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    raw    = _llm_call(llm, _SQL_SYSTEM, query, history)
    parsed = _parse_json_safe(raw)

    if parsed is None or "count_sql" not in parsed or "list_sql" not in parsed:
        logger.error(f"[job_search_agent] LLM 返回格式异常: {raw[:200]}")
        return _error_response("条件解析失败，请重新描述您的需求")

    count_sql = parsed["count_sql"]
    list_sql  = parsed["list_sql"]
    message   = parsed.get("message", "为您搜索匹配职位")

    logger.info(f"[job_search_agent] count_sql: {count_sql}")
    logger.info(f"[job_search_agent] list_sql:  {list_sql}")

    repo  = JobRepo()
    total = repo.execute_count_query(count_sql)
    jobs  = repo.execute_job_query(list_sql)

    if total < 0:
        total = len(jobs)

    if not jobs:
        guide = _llm_call(
            llm,
            _NO_RESULT_SYSTEM,
            f"用户查询：{query}\n搜索条件：{message}",
        )
        return {
            "intent": "job_search",
            "data": {"total": 0, "jobs": [], "message": guide},
            "status": "success",
        }

    return {
        "intent": "job_search",
        "data": {
            "total":   total,
            "jobs":    jobs,
            "message": f"{message}，共找到 {total} 个职位",
        },
        "status": "success",
    }
