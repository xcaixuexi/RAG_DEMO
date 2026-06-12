"""
job_search_agent.py — 求职者职位搜索 Agent（分页版）

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
改进：SQL 生成 Prompt 增加 3 个 Few-Shot 示例，覆盖：
    单条件 / 多条件 AND / 无精确匹配场景
改动：
    - SQL 固定 LIMIT 100（MAX_FETCH），不再由 LLM 决定
    - handle() 返回第 1 页切片 + 分页元信息，同时将全量数据写入 session 缓存
    - 新增 result_type = "jobs"，供翻页接口识别缓存类型

v2 变更：
    - 新增 _is_count_only() 检测：用户只需要总数时跳过 list_sql，节省 DB IO
    - count-only 响应中 pagination 字段为 None，前端不渲染翻页组件
"""

import json
import logging
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
# Prompt（含 Few-Shot 示例）
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
    3. list_sql 必须加 LIMIT 100（固定值，不可更改）
    4. list_sql SELECT 字段固定为：
       id, name, company_name, company_logo, salary, salary_min, salary_max,
       job_exp, education, job_type, job_duty, work_city,
       contact_name, contact_phone, welfare
    5. count_sql 固定为：SELECT COUNT(*) AS total FROM job WHERE ...

只输出 JSON，不输出任何其他文字，格式：
{"count_sql": "...", "list_sql": "...", "message": "..."}

═══════════════════════════════════════════════
Few-Shot 示例
═══════════════════════════════════════════════

【示例1 — 单条件：城市】
用户输入: 深圳有哪些职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND work_city LIKE '%深圳%'",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND work_city LIKE '%深圳%' LIMIT 100",
    "message": "搜索深圳地区所有职位"
}

【示例2 — 多条件 AND：城市 + 岗位 + 薪资】
用户输入: 上海月薪15k以上的产品经理职位
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND work_city LIKE '%上海%' AND name LIKE '%产品经理%' AND salary_min >= 15000",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND work_city LIKE '%上海%' AND name LIKE '%产品经理%' AND salary_min >= 15000 LIMIT 100",
    "message": "搜索上海月薪15k以上的产品经理职位"
}

【示例3 — 多条件 AND：职位类型 + 学历 + 经验】
用户输入: 本科以上学历3年经验的全职前端工程师
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND job_type=0 AND name LIKE '%前端%' AND education='本科' AND job_exp='3-5年'",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND job_type=0 AND name LIKE '%前端%' AND education='本科' AND job_exp='3-5年' LIMIT 100",
    "message": "搜索本科学历3-5年经验的全职前端工程师职位"
}

【示例4 — 职位类型：临时工】
用户输入: 东莞有没有临时工
输出:
{
    "count_sql": "SELECT COUNT(*) AS total FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND work_city LIKE '%东莞%' AND job_type=3",
    "list_sql": "SELECT id, name, company_name, company_logo, salary, salary_min, salary_max, job_exp, education, job_type, job_duty, work_city, contact_name, contact_phone, welfare FROM job WHERE status=1 AND is_delete=0 AND audit_status=1 AND work_city LIKE '%东莞%' AND job_type=3 LIMIT 100",
    "message": "搜索东莞地区临时工职位"
}
═══════════════════════════════════════════════"""

_NO_RESULT_SYSTEM = """你是一个友好的招聘助手。
用户搜索职位无结果，请给出简短、友好的建议，引导用户放宽条件重试。
回复控制在 50 字以内，不用列条目，一段话即可。"""


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
        logger.error(f"[job_search_agent] JSON 解析失败: {e} | 原文: {text[:200]}")
        return None


def _error_response(message: str) -> dict:
    return {"intent": "job_search", "data": {"message": message}, "status": "error"}


def _build_response(page_data: dict, total_db: int, message: str) -> dict:
    """组装统一分页响应格式"""
    return {
        "intent": "job_search",
        "data": {
            "message":     message,
            "jobs":        page_data["items"],
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
    history:    Optional[list[dict]] = None,
    llm = None,
    page_size:  int = DEFAULT_PAGE_SIZE,
) -> dict:
    """
    首次查询入口：LLM 生成 SQL → 查 DB → 缓存全量 → 返回第 1 页。

    Args:
        query:      用户输入
        session_id: 会话 ID，用于写入分页缓存
        history:    多轮对话历史
        llm:        ChatOpenAI 实例
        page_size:  每页条数，默认 20
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
    llm_msg   = parsed.get("message", "为您搜索匹配职位")

    logger.info(f"[job_search_agent] count_sql: {count_sql}")
    logger.info(f"[job_search_agent] list_sql:  {list_sql}")

    repo     = JobRepo()

    # count-only 模式：只查总数，跳过 list_sql，节省 DB IO
    count_only = _is_count_only(query)
    total_db   = repo.execute_count_query(count_sql)

    if count_only:
        if total_db < 0:
            total_db = 0
        return {
            "intent": "job_search",
            "data": {
                "message":    f"{llm_msg}，共找到 {total_db} 个职位",
                "jobs":       [],
                "pagination": None,   # 无分页数据，前端不渲染翻页组件
            },
            "status": "success",
        }

    jobs = repo.execute_job_query(list_sql)     # SQL 已含 LIMIT 100

    if total_db < 0:
        total_db = len(jobs)

    if not jobs:
        guide = _llm_call(llm, _NO_RESULT_SYSTEM, f"用户查询：{query}\n搜索条件：{llm_msg}")
        return {
            "intent": "job_search",
            "data": {
                "message": guide,
                "jobs":    [],
                "pagination": {
                    "page": 1, "page_size": page_size,
                    "total_pages": 0, "fetched": 0, "total_db": 0,
                },
            },
            "status": "success",
        }

    # 写入分页缓存
    save_page_cache(
        session_id  = session_id,
        result_type = RESULT_TYPE,
        items       = jobs,
        total_db    = total_db,
        query       = query,
    )

    # 取第 1 页
    page_data = get_page(session_id, RESULT_TYPE, page=1, page_size=page_size)
    fetched   = len(jobs)

    hint = f"（数据库共 {total_db} 条，已为您加载前 {fetched} 条）" if total_db > fetched else ""
    message = f"{llm_msg}，共找到 {fetched} 个职位{hint}"

    return _build_response(page_data, total_db, message)
