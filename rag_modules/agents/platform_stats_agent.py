"""
platform_stats_agent.py — 平台管理员数据统计 Agent

定位：仅供 admin 角色使用，查询该租户下的平台级聚合统计数据。

内部三级路由：
    Level-1  固定模板匹配（无 LLM）
                ├─ 单模板命中  → 直接执行预设 SQL（支持参数动态注入）
                └─ 多模板命中  → 并行执行多条预设 SQL，合并结果
    Level-2  复杂度信号检测    — 含对比/趋势/分析等分析类词时调用 LLM 生成 SQL
    Level-3  引导语兜底        — 两级均未命中时引导用户换个问法

v2 变更：
    - 固定模板结构改为 {key: {"keywords": [...], "params": [...]}}，新增 params 字段标记支持的动态参数
    - 新增 _extract_params()：从 query 提取城市/行业/时间参数
    - _build_fixed_sql() 支持参数动态注入（城市过滤、时间范围过滤）
    - _match_fixed_template() 拆分为 _match_fixed_templates()，返回所有命中 key 列表
    - 新增 _dedup_keys()：job_count 和 job_active 同时命中时保留 job_count
    - 新增 _execute_multi_templates()：并行执行多条预设 SQL 并合并结果
    - _format_fixed_result() 新增 params 参数，格式化时感知城市/行业上下文
    - handle() 主流程重写：支持单/多模板命中及复杂查询分支
    - _detect_complex_query() 移除时间类信号词，只保留分析/对比类词
    - _LLM_SQL_SYSTEM 末尾追加多问题 → 数组格式说明
    - _handle_llm() 支持 LLM 返回单对象或数组，循环执行并合并结果

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
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 固定模板配置（v2：含 params 字段）
# ─────────────────────────────────────────────

# 每个模板结构：
#   keywords — 触发词列表
#   params   — 该模板支持动态注入的参数名列表（空表示无动态参数）
#              可选值：city / industry / time
_FIXED_TEMPLATES: dict[str, dict] = {
    "company_count": {
        "keywords": ["有多少家", "企业总数", "公司总数", "多少家企业", "多少家公司"],
        "params":   [],
    },
    "company_pending": {
        "keywords": ["待审核企业", "未审核公司", "待审核的企业"],
        "params":   [],
    },
    "company_by_industry": {
        "keywords": ["各行业企业", "行业分布", "按行业", "各个行业"],
        "params":   ["industry"],   # 有行业名 → WHERE 过滤；无行业名 → GROUP BY
    },
    "job_count": {
        "keywords": [
            "有多少个职位", "职位总数", "岗位总数", "多少个职位",
            "发布了多少职位", "发布的职位", "多少个在招", "发布了多少个",  # v2 追问场景
        ],
        "params":   [],
    },
    "job_active": {
        "keywords": [
            "在招职位", "发布中", "已发布职位",
            "发布了多少", "多少个发布", "在招的有多少",  # v2 追问场景
        ],
        "params":   ["city"],   # 支持城市过滤
    },
    "job_pending": {
        "keywords": ["待审核职位", "未审核职位"],
        "params":   [],
    },
    "job_by_city": {
        "keywords": [
            "各城市职位", "城市分布", "按城市", "各个城市",
            "各城市在招", "城市职位分布",
        ],
        "params":   ["city"],   # 有城市名 → WHERE 过滤；无城市名 → GROUP BY
    },
    "apply_count": {
        "keywords": ["报名总数", "报名数", "有多少人报名", "报名情况"],
        "params":   ["time"],   # 支持时间范围过滤
    },
}


# ─────────────────────────────────────────────
# 复杂度信号词
# v2 说明：时间词（今天/本月/上个月等）移出，交给 _extract_params() 结构化处理。
# 只保留"分析/对比/趋势"等需要 LLM 生成复杂 SQL 的场景。
# ─────────────────────────────────────────────

_COMPLEX_SIGNALS: list[str] = [
    "对比", "趋势", "增长", "变化", "同比", "环比",
    "完成率", "转化率", "交叉", "分析",
]


# ─────────────────────────────────────────────
# 引导语（三级均未命中时返回）
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
# v2 末尾追加多问题 → 数组格式说明
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
    4. 不要使用 UNION，多个统计需求请生成多条独立 SQL

若用户问题包含多个独立统计需求，请生成多条独立 SQL，以 JSON 数组形式返回：
[
  {{"sql": "SELECT COUNT(*) FROM company WHERE tenant_id={tenant_id}", "message": "企业总数查询"}},
  {{"sql": "SELECT COUNT(*) FROM job WHERE tenant_id={tenant_id} AND status=1", "message": "在招职位查询"}}
]
若只有一个需求，仍返回单个对象（非数组）：
{{"sql": "...", "message": "一句话说明查询意图"}}

只输出 JSON，不要输出任何其他内容。"""


# ─────────────────────────────────────────────
# 参数提取层（v2 新增）
# ─────────────────────────────────────────────

def _extract_params(query: str) -> dict:
    """
    从 query 中提取动态参数，供固定模板 SQL 动态注入。
    返回 {"city": str|None, "industry": str|None, "time": dict|None}
    time 格式：{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}

    注意：
        城市用 LIKE '%xxx%' 而非精确匹配，兼容"东莞"与"东莞市"两种存储格式。
        行业去后缀（"行"字），避免"制造行业"与数据库"制造业"对不上。
    """
    result: dict = {"city": None, "industry": None, "time": None}
    now = datetime.now()

    # ── 城市提取 ──────────────────────────────────────────
    city_pattern = re.compile(r"([\u4e00-\u9fa5]{2,6}(?:省|市|区|县|地区|自治区))")
    city_match = city_pattern.search(query)
    if city_match:
        result["city"] = city_match.group(1)

    # ── 行业提取 ──────────────────────────────────────────
    # 匹配"XX行业"、"XX产业"、"XX领域"，去除末尾"行"字防止和数据库存储不一致
    industry_pattern = re.compile(r"([\u4e00-\u9fa5]{2,8}(?:行业|产业|领域|行))")
    industry_match = industry_pattern.search(query)
    if industry_match:
        result["industry"] = industry_match.group(1).rstrip("行")

    # ── 时间提取（结构化处理，不再触发 LLM 复杂查询路径）──
    if "今天" in query:
        today = now.strftime("%Y-%m-%d")
        result["time"] = {"start": today, "end": today}

    elif "本周" in query:
        weekday = now.weekday()
        start = (now - timedelta(days=weekday)).strftime("%Y-%m-%d")
        result["time"] = {"start": start, "end": now.strftime("%Y-%m-%d")}

    elif "本月" in query:
        start = now.strftime("%Y-%m-01")
        result["time"] = {"start": start, "end": now.strftime("%Y-%m-%d")}

    elif "上个月" in query or "上月" in query:
        first_of_month    = now.replace(day=1)
        last_month_end    = first_of_month - timedelta(days=1)
        last_month_start  = last_month_end.replace(day=1)
        result["time"] = {
            "start": last_month_start.strftime("%Y-%m-%d"),
            "end":   last_month_end.strftime("%Y-%m-%d"),
        }

    elif "今年" in query:
        result["time"] = {
            "start": now.strftime("%Y-01-01"),
            "end":   now.strftime("%Y-%m-%d"),
        }

    # 匹配"最近N天"
    recent_match = re.compile(r"最近(\d+)天").search(query)
    if recent_match:
        days  = int(recent_match.group(1))
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        result["time"] = {"start": start, "end": now.strftime("%Y-%m-%d")}

    return result


# ─────────────────────────────────────────────
# 多模板匹配与去重（v2 新增）
# ─────────────────────────────────────────────

def _match_fixed_templates(query: str) -> list[str]:
    """
    返回所有命中的模板 key 列表（按 _FIXED_TEMPLATES 定义顺序），未命中返回空列表。
    v2：替换原 _match_fixed_template()（单命中版）。
    """
    text    = query.strip()
    matched = []
    for key, conf in _FIXED_TEMPLATES.items():
        if any(kw in text for kw in conf["keywords"]):
            matched.append(key)
    if matched:
        logger.info(f"[platform_stats] 固定模板命中: '{query}' → {matched}")
    return matched


def _dedup_keys(keys: list[str]) -> list[str]:
    """
    去重规则：
        job_count 和 job_active 同时命中时，保留 job_count（状态分组信息更完整）。
    其余 key 不去重。
    """
    if "job_count" in keys and "job_active" in keys:
        keys = [k for k in keys if k != "job_active"]
    return keys


# ─────────────────────────────────────────────
# 固定 SQL 构建（v2：支持动态参数注入）
# ─────────────────────────────────────────────

def _build_fixed_sql(key: str, tenant_id: int, params: dict) -> Optional[str]:
    """
    根据模板 key、tenant_id 和提取到的参数动态构建 SQL。
    返回 None 时表示参数缺失或配置错误，调用方应降级走 LLM。

    city     用 LIKE '%xxx%' 兼容"东莞"与"东莞市"两种存储格式。
    industry 去后缀后用 LIKE '%xxx%' 兼容"制造业"与"制造"两种存储格式。
    time     转化为 BETWEEN 范围条件（end 日期末尾补 23:59:59）。
    """
    city     = params.get("city")
    industry = params.get("industry")
    time     = params.get("time")

    if key == "company_count":
        return (
            f"SELECT COUNT(*) AS total FROM company "
            f"WHERE tenant_id={tenant_id} AND is_delete=0"
        )

    elif key == "company_pending":
        return (
            f"SELECT COUNT(*) AS total FROM company "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND apply_status=0"
        )

    elif key == "company_by_industry":
        if industry:
            # 有行业名 → 单行业过滤
            return (
                f"SELECT COUNT(*) AS total FROM company "
                f"WHERE tenant_id={tenant_id} AND is_delete=0 "
                f"AND industry LIKE '%{industry}%'"
            )
        else:
            # 无行业名 → 按行业分组汇总
            return (
                f"SELECT industry, COUNT(*) AS total FROM company "
                f"WHERE tenant_id={tenant_id} AND is_delete=0 AND apply_status=1 "
                f"GROUP BY industry ORDER BY total DESC"
            )

    elif key == "job_count":
        sql = (
            f"SELECT status, COUNT(*) AS cnt FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0"
        )
        if time:
            sql += f" AND create_time BETWEEN '{time['start']}' AND '{time['end']} 23:59:59'"
        sql += " GROUP BY status"
        return sql

    elif key == "job_active":
        sql = (
            f"SELECT COUNT(*) AS total FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND status=1 AND audit_status=1"
        )
        if city:
            sql += f" AND work_city LIKE '%{city}%'"
        if time:
            sql += f" AND deploy_time BETWEEN '{time['start']}' AND '{time['end']} 23:59:59'"
        return sql

    elif key == "job_pending":
        return (
            f"SELECT COUNT(*) AS total FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND audit_status=0"
        )

    elif key == "job_by_city":
        if city:
            # 有城市名 → 单城市过滤计数
            return (
                f"SELECT COUNT(*) AS total FROM job "
                f"WHERE tenant_id={tenant_id} AND is_delete=0 "
                f"AND status=1 AND audit_status=1 AND work_city LIKE '%{city}%'"
            )
        else:
            # 无城市名 → 按城市分组汇总
            return (
                f"SELECT work_city, COUNT(*) AS cnt FROM job "
                f"WHERE tenant_id={tenant_id} AND is_delete=0 "
                f"AND status=1 AND audit_status=1 "
                f"GROUP BY work_city ORDER BY cnt DESC LIMIT 20"
            )

    elif key == "apply_count":
        sql = (
            f"SELECT COUNT(*) AS total FROM employees_apply ea "
            f"LEFT JOIN job j ON ea.job_id = j.id "
            f"WHERE j.tenant_id={tenant_id}"
        )
        if time:
            sql += f" AND ea.create_time BETWEEN '{time['start']}' AND '{time['end']} 23:59:59'"
        return sql

    return None


# ─────────────────────────────────────────────
# 结果格式化（v2：新增 params 参数，感知城市/行业上下文）
# ─────────────────────────────────────────────

def _format_fixed_result(
    template_key: str,
    rows:         list[dict],
    params:       Optional[dict] = None,
) -> tuple[str, dict]:
    """
    将固定模板查询结果格式化为 (message, stats_dict)。
    stats_dict 用于响应中的结构化数据字段。
    params 用于在 message 中加入城市/行业/时间上下文描述。
    """
    params = params or {}
    city     = params.get("city") or ""
    industry = params.get("industry") or ""
    time     = params.get("time")
    time_desc = f"{time['start']} 至 {time['end']}" if time else ""

    if not rows:
        return "暂无数据", {}

    if template_key == "company_count":
        total = rows[0].get("total", 0)
        return f"平台共有 {total} 家企业", {"total": total}

    if template_key == "company_pending":
        pending = rows[0].get("total", 0)
        return f"待审核企业共 {pending} 家", {"pending": pending}

    if template_key == "company_by_industry":
        if industry:
            # 单行业过滤，返回计数
            total = rows[0].get("total", 0)
            return f"{industry}行业企业共 {total} 家", {"total": total}
        else:
            # 分组汇总
            parts = [
                f"{r.get('industry', '未知')}（{r.get('total', 0)} 家）"
                for r in rows[:10]
            ]
            return "企业行业分布：" + "、".join(parts), {"rows": rows}

    if template_key == "job_count":
        _status_label = {0: "未审核", 1: "已发布", 2: "不通过", 3: "停止发布"}
        parts = [
            f"{_status_label.get(r.get('status'), '未知')}（{r.get('cnt', 0)} 个）"
            for r in rows
        ]
        prefix = f"{time_desc} " if time_desc else ""
        return f"{prefix}职位状态分布：" + "、".join(parts), {"rows": rows}

    if template_key == "job_active":
        total     = rows[0].get("total", 0)
        city_desc = f"{city}" if city else "平台"
        time_hint = f"（{time_desc}）" if time_desc else ""
        return f"{city_desc}当前在招职位共 {total} 个{time_hint}", {"total": total}

    if template_key == "job_pending":
        total = rows[0].get("total", 0)
        return f"待审核职位共 {total} 个", {"total": total}

    if template_key == "job_by_city":
        if "work_city" in (rows[0] if rows else {}):
            # 分组汇总
            parts = [
                f"{r.get('work_city', '未知')}（{r.get('cnt', 0)} 个）"
                for r in rows[:10]
            ]
            return "各城市在招职位：" + "、".join(parts), {"rows": rows}
        else:
            # 单城市过滤
            total = rows[0].get("total", 0)
            return f"{city}当前在招职位共 {total} 个", {"total": total}

    if template_key == "apply_count":
        total     = rows[0].get("total", 0)
        time_hint = f"（{time_desc}）" if time_desc else ""
        return f"平台总报名记录共 {total} 条{time_hint}", {"total": total}

    return "查询完成", {"rows": rows}


# ─────────────────────────────────────────────
# 多模板并行执行与结果合并（v2 新增）
# ─────────────────────────────────────────────

def _execute_multi_templates(keys: list[str], tenant_id: int, params: dict) -> dict:
    """
    并行执行多个固定模板 SQL，合并结果。
    单条执行失败时跳过（不终止整体），仍返回其余成功结果。

    去重说明：调用前已经过 _dedup_keys()，这里不再二次去重。

    返回：
        {
            "message": "结果1；结果2",
            "stats":   {合并的结构化数据}
        }
        若全部执行失败，返回 {"message": "暂无数据", "stats": {}}
    """
    messages: list[str] = []
    stats:    dict      = {}

    for key in keys:
        sql = _build_fixed_sql(key, tenant_id, params)
        if sql is None:
            logger.warning(f"[platform_stats] 模板 {key} SQL 构建失败，跳过")
            continue

        rows, err = _execute_sql(sql)
        if err:
            logger.error(f"[platform_stats] 模板 {key} SQL 执行失败: {err}")
            continue

        msg, stat = _format_fixed_result(key, rows, params)
        messages.append(msg)

        # 合并 stats，分组类数据用带前缀的 key 区分避免覆盖
        for k, v in stat.items():
            stats_key = f"{key}_{k}" if k in stats else k
            stats[stats_key] = v

    return {
        "message": "；".join(messages) if messages else "暂无数据",
        "stats":   stats,
    }


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _detect_complex_query(query: str) -> bool:
    """
    检测是否为分析/对比类复杂查询（需要 LLM 生成 SQL）。
    v2 说明：时间词已从信号词中移除，由 _extract_params() 结构化处理，
    不再触发 LLM 路径，避免"上个月报名数"这类简单时间过滤被误判为复杂查询。
    """
    return any(sig in query for sig in _COMPLEX_SIGNALS)


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
            keys   = result.keys()
            rows   = [dict(zip(keys, row)) for row in result.fetchall()]
            return rows, None
    except Exception as e:
        logger.error(f"[platform_stats] SQL 执行失败: {e} | SQL: {sql[:200]}")
        return [], str(e)


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


def _guide_response() -> dict:
    """未命中任何规则时返回引导语"""
    return _build_response(_GUIDE_MESSAGE)


def _error_response(message: str) -> dict:
    return {
        "intent": "platform_stats",
        "data":   {"message": message},
        "status": "error",
    }


# ─────────────────────────────────────────────
# LLM 路径（v2：支持单对象/数组两种返回格式）
# ─────────────────────────────────────────────

def _call_llm(query: str, tenant_id: int, llm: ChatOpenAI) -> str:
    """调用 LLM 生成 SQL，返回原始字符串（含 JSON）"""
    system   = _LLM_SQL_SYSTEM.replace("{tenant_id}", str(tenant_id))
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=query),
    ]
    raw = (llm | StrOutputParser()).invoke(messages).strip()
    # 兼容 ```json 围栏
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw   = "\n".join(inner)
    return raw


def _handle_llm(query: str, tenant_id: int, llm: ChatOpenAI) -> dict:
    """
    LLM 路径：调用 LLM 生成 SQL → 执行 → 合并结果。
    支持 LLM 返回单个对象（一个需求）或数组（多个独立需求）。
    每条 SQL 独立做 tenant_id 安全校验，校验失败时整体拒绝。
    """
    logger.info(f"[platform_stats] 复杂查询，调用 LLM 生成 SQL: '{query}'")

    try:
        raw    = _call_llm(query, tenant_id, llm)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"[platform_stats] LLM SQL 解析失败: {e}")
        return _error_response("SQL 生成失败，请换个问法")

    # 统一为列表处理
    items: list[dict] = parsed if isinstance(parsed, list) else [parsed]

    messages: list[str] = []
    rows_all: list[dict] = []

    for item in items:
        sql = item.get("sql", "").strip()
        if not sql:
            continue

        # 安全校验：每条 SQL 都必须包含 tenant_id 过滤
        if not _validate_tenant_id(sql, tenant_id):
            logger.error(f"[platform_stats] LLM 生成 SQL 缺少 tenant_id 过滤: {sql[:100]}")
            return _error_response("查询生成异常，请重试或联系管理员")

        rows, err = _execute_sql(sql)
        if err:
            logger.error(f"[platform_stats] LLM SQL 执行失败: {err}")
            continue

        messages.append(item.get("message", "查询完成"))
        rows_all.extend(rows)

    if not messages:
        return _error_response("查询执行失败，请稍后重试")

    return _build_response("；".join(messages), rows=rows_all)


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query:     str,
    tenant_id: int,
    llm:       Optional[ChatOpenAI] = None,
) -> dict:
    """
    平台统计 Agent 主入口（v2 重写）。

    三级路由流程：
        matched_keys = _dedup_keys(_match_fixed_templates(query))
        has_complex  = _detect_complex_query(query)
        params       = _extract_params(query)

        情况一：未命中任何固定模板
            ├─ has_complex → LLM 路径
            └─ 否则        → 引导语兜底

        情况二：命中模板，但含复杂信号词（对比/趋势/分析等）
            └─ 整句走 LLM（不拆模板，保留完整语义）

        情况三：命中模板，无复杂信号词 → 固定 SQL 路径
            ├─ 模板需要参数 & 参数提取失败 → 降级走 LLM
            ├─ 单模板命中  → 执行单条预设 SQL → 返回
            └─ 多模板命中  → 并行执行所有预设 SQL → 合并结果返回

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

    matched_keys = _dedup_keys(_match_fixed_templates(query))
    has_complex  = _detect_complex_query(query)
    params       = _extract_params(query)

    # ── 情况一：未命中任何固定模板 ────────────────────────────
    if not matched_keys:
        if has_complex:
            return _handle_llm(query, tenant_id, llm)
        logger.debug(f"[platform_stats] 未命中任何规则，返回引导语: '{query}'")
        return _guide_response()

    # ── 情况二：命中模板，但含复杂信号词 → 整句走 LLM ──────────
    if has_complex:
        return _handle_llm(query, tenant_id, llm)

    # ── 情况三：命中模板，无复杂信号词 → 固定 SQL 路径 ──────────
    # 检查每个命中模板：若需要参数且参数提取失败，则降级走 LLM
    for key in matched_keys:
        template_params   = _FIXED_TEMPLATES[key]["params"]
        needs_param       = len(template_params) > 0
        param_extracted   = any(params.get(p) for p in template_params)
        if needs_param and not param_extracted:
            logger.info(
                f"[platform_stats] 模板 {key} 需要参数 {template_params} "
                f"但提取失败，降级走 LLM"
            )
            return _handle_llm(query, tenant_id, llm)

    if len(matched_keys) == 1:
        # ── 单模板命中：原有逻辑不变 ───────────────────────────
        key  = matched_keys[0]
        sql  = _build_fixed_sql(key, tenant_id, params)
        if sql is None:
            return _handle_llm(query, tenant_id, llm)

        rows, err = _execute_sql(sql)
        if err:
            return _error_response(f"数据查询失败，请稍后重试（{err[:50]}）")

        message, stats = _format_fixed_result(key, rows, params)
        # 分组类（含 rows 键）和单值类（含 total/pending 等键）分别返回
        if "rows" in stats:
            return _build_response(message, rows=stats["rows"])
        return _build_response(message, stats=stats)

    else:
        # ── 多模板命中：并行执行，合并结果 ─────────────────────
        logger.info(f"[platform_stats] 多模板并行执行: {matched_keys}")
        result = _execute_multi_templates(matched_keys, tenant_id, params)
        return _build_response(result["message"], stats=result["stats"] or None)
