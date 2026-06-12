"""
platform_stats_agent.py — 平台管理员数据统计 Agent

定位：仅供 admin 角色使用，查询该租户下的平台级聚合统计数据。

内部二级路由：
    Level-1  固定模板匹配（无 LLM）— 覆盖 8 个高频统计维度，毫秒级响应
    Level-2  复杂度信号检测         — 含时间范围/对比/趋势等词时，调用 LLM 生成 SQL
    Level-3  引导语兜底             — 两级均未命中时，引导用户换个问法

API 响应统一格式：
    {
        "intent": "platform_stats",
        "data": {
            "message": "...",
            "stats": {...}   # 固定模板时的结构化数据（可选）
            "rows":  [...]   # LLM 生成 SQL 的原始查询结果（可选）
        },
        "status": "success"
    }
"""

import json
import logging
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 固定模板配置
# ─────────────────────────────────────────────

# key → 触发关键词列表
_FIXED_TEMPLATES: dict[str, list[str]] = {
    "company_count":       ["有多少家", "企业总数", "公司总数", "多少家企业", "多少家公司"],
    "company_pending":     ["待审核企业", "未审核公司", "待审核的企业"],
    "company_by_industry": ["各行业企业", "行业分布", "按行业"],
    "job_count":           ["有多少个职位", "职位总数", "岗位总数", "多少个职位"],
    "job_active":          ["在招职位", "发布中", "已发布职位"],
    "job_pending":         ["待审核职位", "未审核职位"],
    "job_by_city":         ["各城市职位", "城市分布", "按城市"],
    "apply_count":         ["报名总数", "报名数", "有多少人报名", "报名情况"],
}

# ─────────────────────────────────────────────
# 复杂度信号词（命中时走 LLM 生成 SQL）
# ─────────────────────────────────────────────

_COMPLEX_SIGNALS: list[str] = [
    "上个月", "最近30天", "今年", "上周", "本季度",
    "对比", "趋势", "增长", "变化", "同比", "环比",
    "按", "分组", "交叉", "完成率", "转化率",
]

# ─────────────────────────────────────────────
# 引导语（两级均未命中时返回）
# ─────────────────────────────────────────────

_GUIDE_MESSAGE = (
    "抱歉，我没能理解您的统计需求。您可以这样问我：\n"
    '· \u201c平台现在有多少家企业\u201d\n'
    '· \u201c待审核的职位有几个\u201d\n'
    '· \u201c各城市职位分布情况\u201d\n'
    '· \u201c平台总报名人数\u201d\n'
    "如果是更复杂的统计需求，请尽量描述清楚时间范围和统计维度。"
)

# ─────────────────────────────────────────────
# LLM Prompt（复杂查询）
# ─────────────────────────────────────────────

_LLM_SQL_SYSTEM = """\
你是平台数据统计助手，根据管理员的自然语言需求，生成查询 SQL。

可查询的表（必须在白名单内）：
    company         — 企业信息表
    job             — 职位信息表
    employees_apply — 员工报名表

数据隔离规则（最高优先级，不可违反）：
    所有查询必须在 WHERE 中包含对应主表的 tenant_id = {tenant_id} 条件
    跨表 JOIN 时，以 company.tenant_id = {tenant_id} 为主过滤条件

固定规则：
    1. 只生成 SELECT 语句
    2. 聚合查询（含 GROUP BY）无需 LIMIT；列表查询加 LIMIT 200
    3. 条件不确定时宁可不加，不要强行猜测
    4. 输出格式为 JSON，不要输出其他内容

只输出 JSON，格式：
{{"sql": "...", "message": "一句话说明查询意图"}}"""


# ─────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────

def _match_fixed_template(query: str) -> Optional[str]:
    """固定模板匹配层，命中返回模板 key，未命中返回 None"""
    text = query.strip()
    for key, keywords in _FIXED_TEMPLATES.items():
        if any(kw in text for kw in keywords):
            logger.info(f"[platform_stats] 固定模板命中: '{query}' → {key}")
            return key
    return None


def _detect_complex_query(query: str) -> bool:
    """检测是否为复杂查询（含时间范围/对比/趋势等信号词）"""
    return any(sig in query for sig in _COMPLEX_SIGNALS)


def _build_fixed_sql(template_key: str, tenant_id: int) -> str:
    """根据固定模板 key 和 tenant_id 构造预设 SQL"""
    sqls = {
        "company_count": (
            f"SELECT COUNT(*) AS total FROM company "
            f"WHERE tenant_id={tenant_id} AND is_delete=0"
        ),
        "company_pending": (
            f"SELECT COUNT(*) AS total FROM company "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND apply_status=0"
        ),
        "company_by_industry": (
            f"SELECT industry, COUNT(*) AS cnt FROM company "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND apply_status=1 "
            f"GROUP BY industry ORDER BY cnt DESC"
        ),
        "job_count": (
            f"SELECT status, COUNT(*) AS cnt FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 GROUP BY status"
        ),
        "job_active": (
            f"SELECT COUNT(*) AS total FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND status=1 AND audit_status=1"
        ),
        "job_pending": (
            f"SELECT COUNT(*) AS total FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND audit_status=0"
        ),
        "job_by_city": (
            f"SELECT work_city, COUNT(*) AS cnt FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND status=1 AND audit_status=1 "
            f"GROUP BY work_city ORDER BY cnt DESC LIMIT 20"
        ),
        "apply_count": (
            f"SELECT COUNT(*) AS total FROM employees_apply ea "
            f"LEFT JOIN job j ON ea.job_id = j.id "
            f"WHERE j.tenant_id={tenant_id}"
        ),
    }
    return sqls.get(template_key, "")


def _execute_sql(sql: str) -> tuple[list[dict], Optional[str]]:
    """
    执行 SQL，返回 (rows, error_message)。
    rows 为 list[dict]，error_message 不为 None 表示出错。
    """
    from sqlalchemy import text
    from db.mysql_client import MySQLClient

    try:
        db = MySQLClient.get_instance()
        with db._session() as session:
            result = session.execute(text(sql))
            keys = result.keys()
            rows = [dict(zip(keys, row)) for row in result.fetchall()]
            return rows, None
    except Exception as e:
        logger.error(f"[platform_stats] SQL 执行失败: {e} | SQL: {sql[:200]}")
        return [], str(e)


def _format_fixed_result(template_key: str, rows: list[dict]) -> tuple[str, dict]:
    """
    将固定模板查询结果格式化为 (message, stats_dict)。
    stats_dict 用于响应中的结构化数据字段。
    """
    if not rows:
        return "暂无数据", {}

    if template_key == "company_count":
        total = rows[0].get("total", 0)
        return f"平台共有 {total} 家企业", {"total": total}

    if template_key == "company_pending":
        pending = rows[0].get("total", 0)
        return f"待审核企业共 {pending} 家", {"pending": pending}

    if template_key == "company_by_industry":
        parts = [f"{r.get('industry', '未知')}（{r.get('cnt', 0)} 家）" for r in rows[:10]]
        return "企业行业分布：" + "、".join(parts), {"rows": rows}

    if template_key == "job_count":
        _status_label = {0: "未审核", 1: "已发布", 2: "不通过", 3: "停止发布"}
        parts = [f"{_status_label.get(r.get('status'), '未知')}（{r.get('cnt', 0)} 个）" for r in rows]
        return "职位状态分布：" + "、".join(parts), {"rows": rows}

    if template_key == "job_active":
        total = rows[0].get("total", 0)
        return f"平台当前在招职位共 {total} 个", {"total": total}

    if template_key == "job_pending":
        total = rows[0].get("total", 0)
        return f"待审核职位共 {total} 个", {"total": total}

    if template_key == "job_by_city":
        parts = [f"{r.get('work_city', '未知')}（{r.get('cnt', 0)} 个）" for r in rows[:10]]
        return "各城市在招职位：" + "、".join(parts), {"rows": rows}

    if template_key == "apply_count":
        total = rows[0].get("total", 0)
        return f"平台总报名人数共 {total} 条", {"total": total}

    return "查询完成", {"rows": rows}


def _llm_generate_sql(
    query: str,
    tenant_id: int,
    llm: ChatOpenAI,
) -> Optional[dict]:
    """调用 LLM 生成复杂统计 SQL，返回解析后的 dict 或 None"""
    # 将 tenant_id 填入 system prompt（避免模板花括号冲突，手动替换）
    system = _LLM_SQL_SYSTEM.replace("{tenant_id}", str(tenant_id))
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=query),
    ]
    try:
        raw = (llm | StrOutputParser()).invoke(messages).strip()
        # 兼容 ```json 围栏
        cleaned = raw
        if cleaned.startswith("```"):
            lines   = cleaned.splitlines()
            inner   = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            cleaned = "\n".join(inner)
        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"[platform_stats] LLM SQL 生成失败: {e}")
        return None


def _validate_tenant_id(sql: str, tenant_id: int) -> bool:
    """校验 SQL 中是否包含 tenant_id 过滤，防止数据越权"""
    return bool(
        re.compile(rf"tenant_id\s*=\s*{tenant_id}", re.IGNORECASE).search(sql)
    )


def _build_response(message: str, stats: Optional[dict] = None, rows: Optional[list] = None) -> dict:
    data: dict = {"message": message}
    if stats:
        data["stats"] = stats
    if rows is not None:
        data["rows"] = rows
    return {
        "intent": "platform_stats",
        "data":   data,
        "status": "success",
    }


def _error_response(message: str) -> dict:
    return {
        "intent": "platform_stats",
        "data":   {"message": message},
        "status": "error",
    }


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query:     str,
    tenant_id: int,
    llm:       Optional[ChatOpenAI] = None,
) -> dict:
    """
    平台统计 Agent 主入口。

    内部二级路由：
        Level-1  固定模板匹配 — 命中则直接执行预设 SQL，不调用 LLM
        Level-2  复杂度检测   — 含信号词时调用 LLM 生成 SQL 执行
        Level-3  引导语兜底   — 两级均未命中时返回引导语

    Args:
        query:     用户输入
        tenant_id: 当前 admin 关联的租户 ID（由 ChatController 强制注入）
        llm:       ChatOpenAI 实例，由 ChatController 从 Supervisor 传入

    Returns:
        统一响应字典
    """
    if llm is None:
        logger.error("[platform_stats] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    # ── Level-1：固定模板匹配 ─────────────────
    template_key = _match_fixed_template(query)
    if template_key:
        sql = _build_fixed_sql(template_key, tenant_id)
        rows, err = _execute_sql(sql)
        if err:
            return _error_response(f"数据查询失败，请稍后重试（{err[:50]}）")
        message, stats = _format_fixed_result(template_key, rows)
        # 分组类返回 rows 结构化数据，单值类返回 stats
        if "rows" in stats:
            return _build_response(message, rows=stats["rows"])
        return _build_response(message, stats=stats)

    # ── Level-2：复杂度检测 → LLM 生成 SQL ───
    if _detect_complex_query(query):
        logger.info(f"[platform_stats] 复杂查询，调用 LLM 生成 SQL: '{query}'")
        parsed = _llm_generate_sql(query, tenant_id, llm)

        if parsed is None or "sql" not in parsed:
            logger.error("[platform_stats] LLM SQL 生成失败或格式异常")
            return _error_response("复杂统计查询解析失败，请尝试换个描述方式。")

        sql     = parsed["sql"]
        llm_msg = parsed.get("message", "统计查询完成")

        # 安全校验：必须包含 tenant_id 过滤
        if not _validate_tenant_id(sql, tenant_id):
            logger.error(f"[platform_stats] LLM 生成 SQL 缺少 tenant_id 过滤: {sql[:100]}")
            return _error_response("查询生成异常，请重试或联系管理员")

        rows, err = _execute_sql(sql)
        if err:
            return _error_response(f"数据查询失败，请稍后重试（{err[:50]}）")

        return _build_response(llm_msg, rows=rows)

    # ── Level-3：引导语兜底 ───────────────────
    logger.debug(f"[platform_stats] 未命中任何规则，返回引导语: '{query}'")
    return _build_response(_GUIDE_MESSAGE)
