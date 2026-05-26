"""
match_agent.py — 职位匹配 Agent

两级业务逻辑：
    求职者（jobseeker）：
        用户描述需求 → LLM 提取条件生成 SQL → 查询 job 表 → 返回职位列表
        无结果时 → LLM 生成友好引导语

    招聘者（recruiter）：
        用户描述职位 → LLM 提取 job_id 或职位名 → 查询 employees_apply → 返回候选人列表

安全约定：
    LLM 生成的 SQL 经 job_repo.execute_job_query 校验，
    只允许 SELECT，禁止任何写操作关键词。
"""

import json
import logging
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from db.repositories.job_repo import JobRepo

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt 定义
# ─────────────────────────────────────────────

# 求职者：提取查询条件，生成 SQL
_JOBSEEKER_SQL_SYSTEM = """你是一个招聘数据库查询助手。
根据用户的自然语言需求，生成一条查询 job 表的 SQL 语句。

表名：job
可用查询字段：
    name        VARCHAR  职位名称，用 LIKE 模糊匹配
    work_city   VARCHAR  工作城市，精确匹配
    salary      VARCHAR  薪资范围，值域：面议/3k以下/3k-5k/5k-8k/8k-12k/12k-15k/15k-20k/20k以上
    job_exp     VARCHAR  工作经验，值域：不限/应届生/3年及以下/3-5年/5-10年/10年以上
    education   VARCHAR  学历要求，值域：不限/大专/本科/硕士/博士
    job_type    TINYINT  职位类型：0全职 1就业 2实习 3临时工

固定规则：
    1. WHERE 条件必须包含 status=1 AND is_delete=0 AND audit_status=1
    2. 条件不确定时宁可不加，不要强行猜测
    3. 结果必须加 LIMIT 20
    4. SELECT 字段固定为：id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare

只输出 JSON，不输出任何其他文字，格式：
{{
    "sql": "完整的 SELECT 语句",
    "message": "一句话说明搜索意图，如：为您搜索深圳3-5年经验的Python开发职位"
}}"""

# 求职者：无结果时引导
_NO_RESULT_SYSTEM = """你是一个友好的招聘助手。
用户搜索职位无结果，请给出简短、友好的建议，引导用户放宽条件重试。
回复控制在 50 字以内，不用列条目，一段话即可。"""

# 招聘者：提取 job_id 或职位名
_RECRUITER_EXTRACT_SYSTEM = """你是一个招聘数据库查询助手。
根据招聘者的问题，提取他想查询的职位信息。

只输出 JSON，格式：
{{
    "job_id": null,          // 若用户明确提到职位ID，填入数字；否则填 null
    "job_name": "职位名称",  // 若用户提到职位名称，填入；否则填 null
    "message": "一句话说明查询意图"
}}

注意：job_id 和 job_name 至少有一个不为 null。"""


# ─────────────────────────────────────────────
# LLM 工具函数
# ─────────────────────────────────────────────

def _llm_call(llm: ChatOpenAI, system: str, user_content: str) -> str:
    """封装单次 LLM 调用，返回字符串"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human",  "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_content}).strip()


def _parse_json_safe(text: str) -> Optional[dict]:
    """安全解析 LLM 返回的 JSON，兼容 ```json 围栏"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[match_agent] JSON 解析失败: {e} | 原文: {text[:200]}")
        return None


# ─────────────────────────────────────────────
# 业务分支
# ─────────────────────────────────────────────

def _handle_jobseeker(query: str, llm: ChatOpenAI, repo: JobRepo) -> dict:
    """
    求职者流程：
        1. LLM 提取条件生成 SQL
        2. 执行 SQL 查询 job 表
        3. 有结果 → 返回职位列表
           无结果 → LLM 生成引导语
    """
    # Step 1：LLM 生成 SQL
    raw = _llm_call(llm, _JOBSEEKER_SQL_SYSTEM, query)
    parsed = _parse_json_safe(raw)

    if parsed is None or "sql" not in parsed:
        logger.error("[match_agent] LLM 未返回有效 SQL JSON")
        return _error_response("条件解析失败，请重新描述您的需求")
    print(parsed)

    sql     = parsed["sql"]
    hint    = parsed.get("message", "正在为您搜索匹配职位")

    logger.info(f"[match_agent] 生成 SQL: {sql}")

    # Step 2：执行查询（repo 内部已做 SQL 安全校验）
    jobs = repo.execute_job_query(sql)

    # Step 3：有结果直接返回
    if jobs:
        total = len(jobs)
        return {
            "intent": "job_match",
            "data": {
                "jobs":    jobs,
                "total":   total,
                "message": f"{hint}，为您找到 {total} 个匹配职位",
            },
            "status": "success",
        }

    # Step 4：无结果，LLM 生成引导语
    guide = _llm_call(
        llm,
        _NO_RESULT_SYSTEM,
        f"用户查询：{query}\n搜索条件：{hint}",
    )
    return {
        "intent": "job_match",
        "data": {
            "jobs":    [],
            "total":   0,
            "message": guide,
        },
        "status": "success",
    }


def _handle_recruiter(query: str, llm: ChatOpenAI, repo: JobRepo) -> dict:
    """
    招聘者流程：
        1. LLM 提取 job_id 或职位名
        2. 查询 employees_apply 返回候选人列表
    """
    # Step 1：LLM 提取查询意图
    raw = _llm_call(llm, _RECRUITER_EXTRACT_SYSTEM, query)
    parsed = _parse_json_safe(raw)

    if parsed is None:
        return _error_response("意图解析失败，请描述您想查询哪个职位的候选人")

    job_id   = parsed.get("job_id")
    job_name = parsed.get("job_name")
    hint     = parsed.get("message", "查询候选人")

    # Step 2：查询候选人
    if job_id:
        candidates = repo.get_candidates_by_job(int(job_id))
    elif job_name:
        candidates = repo.get_candidates_by_job_name(job_name)
    else:
        return _error_response("未能识别职位信息，请说明职位 ID 或职位名称")

    total = len(candidates)
    message = f"{hint}，该职位共有 {total} 名候选人" if total else "暂无候选人报名记录"

    return {
        "intent": "job_match",
        "data": {
            "candidates": candidates,
            "total":      total,
            "message":    message,
        },
        "status": "success",
    }


def _error_response(message: str) -> dict:
    return {
        "intent": "job_match",
        "data":   {"message": message},
        "status": "error",
    }


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query:     str,
    user_role: str = "jobseeker",
    history:   Optional[list[dict]] = None,
    llm:       Optional[ChatOpenAI] = None,
) -> dict:
    """
    职位匹配 Agent 主入口。

    Args:
        query:     用户输入
        user_role: "jobseeker" → 查职位；"recruiter" → 查候选人
        history:   多轮历史（预留，暂未使用）
        llm:       ChatOpenAI 实例，由 ChatController 从 Supervisor 传入

    Returns:
        统一响应字典
    """
    history = history or []

    if llm is None:
        logger.error("[match_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    repo = JobRepo()

    if user_role == "recruiter":
        return _handle_recruiter(query, llm, repo)
    else:
        return _handle_jobseeker(query, llm, repo)
