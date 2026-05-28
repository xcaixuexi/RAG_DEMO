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
    work_city   VARCHAR  工作城市，用 LIKE 模糊匹配
    salary_min  INT      最低薪资（元/月）
    salary_max  INT      最高薪资（元/月）
    salary      VARCHAR  薪资范围，值域：面议/3k以下/3k-5k/5k-8k/8k-12k/12k-15k/15k-20k/20k以上
    job_exp     VARCHAR  工作经验，值域：不限/应届生/3年及以下/3-5年/5-10年/10年以上
    education   VARCHAR  学历要求，值域：不限/大专/本科/硕士/博士
    job_type    TINYINT  职位类型：0全职 1就业 2实习 3临时工

查询类型区分：
    - 列表查询：用户想看职位列表、找职位、推荐职位
      → 使用固定 SELECT 字段 + LIMIT 20
      → type 字段填 "list"
    - 统计查询：用户询问数量、有多少、统计等
      → 只输出 SELECT COUNT(*) AS total FROM job WHERE ...（不加 LIMIT）
      → type 字段填 "count"

多条件规则：
    - 用户提到多个条件时，所有条件用 AND 连接，不要只取其中一个
    - 例如用户说"xx城市的电工招聘情况"
      → WHERE work_city LIKE '%xx%' AND name LIKE '%电工%'
    - 例如用户说"深圳地区5年经验的开发职位"
      → WHERE work_city='深圳' AND job_exp='5-10年'
    - 条件之间是并列关系，不要丢弃任何一个明确提到的条件

固定规则：
    1. WHERE 条件必须包含 status=1 AND is_delete=0 AND audit_status=1
    2. 条件不确定时宁可不加，不要强行猜测
    3. 列表查询 SELECT 字段固定为：id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare

只输出 JSON，不输出任何其他文字，格式：
{{
    "type": "list",
    "sql": "完整的 SELECT 语句",
    "message": "一句话说明搜索意图，如：为您搜索深圳3-5年经验的Python开发职位"
}}"""

# 求职者：无结果时引导
_NO_RESULT_SYSTEM = """你是一个友好的招聘助手。
用户搜索职位无结果，请给出简短、友好的建议，引导用户放宽条件重试。
回复控制在 50 字以内，不用列条目，一段话即可。"""

# 招聘者：三表 JOIN 版本，支持列表和统计两种查询
_RECRUITER_EXTRACT_SYSTEM = """你是一个招聘数据库查询助手。
根据招聘者的问题，提取查询条件生成 SQL。

涉及的表：
    employees_apply（报名表，别名 ea）
    job（职位表，别名 j）
    company（企业表，别名 c）

关联关系：
    ea.job_id = j.id
    ea.company_id = c.id

可用查询条件：
    职位名称：j.name LIKE '%xxx%' 或 ea.job_name LIKE '%xxx%'
    公司名称：c.name LIKE '%xxx%' 或 ea.work_company_name LIKE '%xxx%'
    审核状态：ea.status（1审核中 2未通过 3在职 4已离职 5报名取消）
    平台审核：ea.audit_type（1待审核 2录用 3不适合）
    期望薪资：ea.expected_salary
    报名方式：ea.emp_way（0自主 1代替）

查询类型区分：
    - 列表查询：查看候选人列表、报名记录
      → 使用固定 SELECT 字段 + LIMIT 50 + ORDER BY ea.create_time DESC
      → type 字段填 "list"
    - 统计查询：询问报名人数、有多少候选人、统计等
      → SELECT COUNT(*) AS total FROM employees_apply ea LEFT JOIN job j ON ea.job_id = j.id LEFT JOIN company c ON ea.company_id = c.id WHERE ...
      → type 字段填 "count"

多条件规则：
    - 用户提到多个条件时，所有条件用 AND 连接，不要只取其中一个
    - 例如用户说"xxx公司的Python开发报名情况"
      → WHERE ea.company_id对应条件 AND ea.job_id对应条件
    - 例如用户说"深圳地区5年经验的开发职位"
      → WHERE work_city='深圳' AND job_exp='5-10年'
    - 条件之间是并列关系，不要丢弃任何一个明确提到的条件

固定规则（列表查询）：
    1. ea.status != 5（过滤已取消报名）
    2. LIMIT 50
    3. ORDER BY ea.create_time DESC

列表查询 SELECT 固定字段：
    ea.id as apply_id,
    ea.user_id,
    ea.resume_id,
    ea.job_id,
    COALESCE(j.name, ea.job_name) as job_name,
    ea.company_id,
    COALESCE(c.name, ea.work_company_name) as company_name,
    ea.expected_salary,
    ea.status,
    ea.audit_type,
    ea.emp_way,
    ea.create_time,
    ea.audit_time,
    ea.cancel_time,
    ea.remark,
    ea.reason

只输出 JSON，格式：
{{
    "type": "list",
    "sql": "完整 SELECT 语句",
    "message": "一句话说明查询意图"
}}"""


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
        1. LLM 提取条件生成 SQL，返回 type（list/count）
        2. count → execute_count_query，返回数量
           list  → execute_job_query，返回职位列表；无结果时 LLM 生成引导语
    """
    raw = _llm_call(llm, _JOBSEEKER_SQL_SYSTEM, query)
    parsed = _parse_json_safe(raw)

    if parsed is None or "sql" not in parsed:
        logger.error("[match_agent] LLM 未返回有效 SQL JSON")
        return _error_response("条件解析失败，请重新描述您的需求")

    sql       = parsed["sql"]
    hint      = parsed.get("message", "正在为您搜索匹配职位")
    query_type = parsed.get("type", "list")

    logger.info(f"[match_agent] 求职者 SQL (type={query_type}): {sql}")

    # ── 统计查询分支 ──────────────────────────
    if query_type == "count":
        total = repo.execute_count_query(sql)
        if total < 0:
            return _error_response("统计查询失败，请稍后重试")
        return {
            "intent": "job_match",
            "data": {
                "total":   total,
                "message": f"{hint}，共有 {total} 个符合条件的职位",
            },
            "status": "success",
        }

    # ── 列表查询分支 ──────────────────────────
    jobs = repo.execute_job_query(sql)

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

    # 无结果，LLM 生成引导语
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
        1. LLM 根据自然语言生成三表 JOIN SQL，返回 type（list/count）
        2. count → execute_count_query，返回报名人数
           list  → execute_apply_query，返回候选人完整列表
    """
    raw = _llm_call(llm, _RECRUITER_EXTRACT_SYSTEM, query)
    parsed = _parse_json_safe(raw)

    if parsed is None or "sql" not in parsed:
        return _error_response("意图解析失败，请描述您想查询哪个职位的候选人")

    sql        = parsed["sql"]
    hint       = parsed.get("message", "查询候选人")
    query_type = parsed.get("type", "list")

    logger.info(f"[match_agent] 招聘者 SQL (type={query_type}): {sql}")

    # ── 统计查询分支 ──────────────────────────
    if query_type == "count":
        total = repo.execute_count_query(sql)
        if total < 0:
            return _error_response("统计查询失败，请稍后重试")
        return {
            "intent": "job_match",
            "data": {
                "total":   total,
                "message": f"{hint}，共有 {total} 条报名记录",
            },
            "status": "success",
        }

    # ── 列表查询分支 ──────────────────────────
    candidates = repo.execute_apply_query(sql)
    total      = len(candidates)
    message    = f"{hint}，共找到 {total} 条报名记录" if total else "暂无符合条件的报名记录"

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
