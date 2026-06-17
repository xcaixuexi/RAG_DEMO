"""
platform_stats_agent.py — 平台管理员数据统计 Agent

定位：仅供 admin 角色使用，查询该租户下的平台级聚合统计数据。

内部路由（v3 重构后）：
    时间词前置检测    — 含"今天/本月/上个月"等时间词时直接走 LLM，
                        避免固定 SQL 丢失时间条件导致结果不正确
    固定模板匹配      — 5 个完全无参数的高频统计查询，命中则直接执行预设 SQL
        ├─ 单模板命中  → 执行对应预设 SQL，返回结构化结果
        └─ 多模板命中  → 并行执行，合并结果
    分析词检测        — 命中固定模板但含"对比/趋势/分析/分布"等分析词时走 LLM
    LLM 兜底          — 未命中固定模板时一律走 LLM，不返回引导语

v1（初始版本）：
    - 8 个固定模板，支持参数提取（城市/行业/时间动态注入）
    - _detect_complex_query()：时间词 + 分析词均触发 LLM
    - 未命中时返回引导语

v2 变更：
    - 固定模板扩充关键词，新增追问场景词
    - _match_fixed_template() → _match_fixed_templates()，返回所有命中 key 列表
    - 新增 _dedup_keys()：job_count + job_active 同时命中时保留 job_count
    - _extract_params() 提取城市/行业/时间，_build_fixed_sql() 支持参数动态注入
    - _format_fixed_result() 感知城市/行业/时间上下文
    - _execute_multi_templates() 并行执行多条预设 SQL
    - _detect_complex_query() 移除时间类词，只保留分析对比类词
    - _LLM_SQL_SYSTEM 末尾追加多需求 → JSON 数组格式说明
    - _handle_llm() 支持 LLM 返回单对象或数组

v3 重构：
    - 固定模板精简为 5 个完全无参数的模板，移除 job_active / job_by_city / company_by_industry
    - 删除 _extract_params()、_COMPLEX_SIGNALS、_GUIDE_MESSAGE、_guide_response()
    - 删除所有函数签名中的 params 参数
    - _detect_complex_query() 改名为 _has_analysis_signals()，职责收窄为"检测分析对比词"
    - 新增 _TIME_SIGNALS 和时间词前置检测（放在模板匹配之前，逻辑独立）
    - handle() 主流程简化：时间词 → LLM；固定模板+分析词 → LLM；固定模板 → 预设SQL；其余 → LLM
    - _build_fixed_sql() 移除所有参数注入，每个模板只有一条固定 SQL
    - _format_fixed_result() 移除 params 参数
    - _execute_multi_templates() 移除 params 参数

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
# 固定模板配置（v3：精简为 5 个完全无参数的模板）
#
# 移除的模板（v3）：
#   job_active       — 需要城市参数，交给 LLM 处理更准确
#   job_by_city      — 需要城市参数，交给 LLM 处理更准确
#   company_by_industry — 需要行业参数，交给 LLM 处理更准确
# ─────────────────────────────────────────────

# 每个模板结构：
#   keywords — 触发词列表
#   params   — 该模板支持动态注入的参数名列表（空表示无动态参数）
_FIXED_TEMPLATES: dict[str, dict] = {
    "company_count": {
        "keywords": ["有多少家", "企业总数", "公司总数", "多少家企业", "多少家公司"],
        "params":   [],
    },
    "company_pending": {
        "keywords": ["待审核企业", "未审核公司", "待审核的企业"],
        "params":   [],
    },
    "job_count": {
        "keywords": [
            "有多少个职位", "职位总数", "岗位总数", "多少个职位",
            "发布了多少职位", "发布的职位", "多少个在招", "发布了多少个",
        ],
        "params":   [],
    },
    "job_pending": {
        "keywords": ["待审核职位", "未审核职位"],
        "params":   [],
    },
    "apply_count": {
        "keywords": ["报名总数", "报名数", "有多少人报名", "报名情况", "报名总量"],
        "params":   [],
    },
}


# ─────────────────────────────────────────────
# 时间词前置检测（v3 新增）
#
# 含时间词时直接走 LLM，不进入固定模板匹配。
# 原因："上个月报名总数"会命中 apply_count，但固定 SQL 无时间条件，结果不正确。
# ─────────────────────────────────────────────

_TIME_SIGNALS: list[str] = [
    "今天", "本周", "本月", "上个月", "上月", "今年",
    "最近", "昨天", "上周", "去年",
]


# ─────────────────────────────────────────────
# 分析信号词（v3：原 _COMPLEX_SIGNALS 改名并精简）
#
# 只保留"对比/趋势/分析"等真正需要 LLM 复杂 SQL 的场景。
# 时间词已移出，由 _TIME_SIGNALS 前置处理。
# ─────────────────────────────────────────────

_ANALYSIS_SIGNALS: list[str] = [
    "对比", "趋势", "增长", "变化", "同比", "环比",
    "完成率", "转化率", "交叉", "分析", "分布", "排名",
]


def _has_analysis_signals(query: str) -> bool:
    """
    检测是否含分析对比类词（原 _detect_complex_query，v3 改名）。
    职责收窄为"检测分析对比词"，时间词检测已移到 handle() 前置。
    """
    return any(sig in query for sig in _ANALYSIS_SIGNALS)


# ─────────────────────────────────────────────
# LLM Prompt（复杂查询 / LLM 兜底）
# ─────────────────────────────────────────────

_LLM_SQL_SYSTEM = """\
你是平台数据统计助手，根据管理员的自然语言需求，生成查询 SQL。

可查询的表（必须在白名单内）：
    company         — 企业信息表
    job             — 职位信息表
    employees_apply — 员工报名表

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
# 多模板匹配与去重
# ─────────────────────────────────────────────

def _match_fixed_templates(query: str) -> list[str]:
    """
    返回所有命中的模板 key 列表（按 _FIXED_TEMPLATES 定义顺序），未命中返回空列表。
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
    v3 说明：job_active 已从固定模板移除，此函数保留以防未来重新加入时的防御处理。
    """
    if "job_count" in keys and "job_active" in keys:
        keys = [k for k in keys if k != "job_active"]
    return keys


# ─────────────────────────────────────────────
# 固定 SQL 构建（v3：移除所有参数注入，每个模板只有一条固定 SQL）
# ─────────────────────────────────────────────

def _build_fixed_sql(key: str, tenant_id: int) -> Optional[str]:
    """
    根据模板 key 和 tenant_id 构建固定预设 SQL。
    v3 变更：移除 params 参数，不再做城市/行业/时间动态注入。
    含动态条件的查询（城市、行业、时间过滤）一律由 LLM 处理。
    """
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

    elif key == "job_count":
        return (
            f"SELECT status, COUNT(*) AS cnt FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 "
            f"GROUP BY status"
        )

    elif key == "job_pending":
        return (
            f"SELECT COUNT(*) AS total FROM job "
            f"WHERE tenant_id={tenant_id} AND is_delete=0 AND audit_status=0"
        )

    elif key == "apply_count":
        return (
            f"SELECT COUNT(*) AS total FROM employees_apply ea "
            f"LEFT JOIN job j ON ea.job_id = j.id "
            f"WHERE j.tenant_id={tenant_id}"
        )

    return None


# ─────────────────────────────────────────────
# 结果格式化（v3：移除 params 参数，不再感知城市/行业/时间上下文）
# ─────────────────────────────────────────────

def _format_fixed_result(template_key: str, rows: list[dict]) -> tuple[str, dict]:
    """
    将固定模板查询结果格式化为 (message, stats_dict)。
    v3 变更：移除 params 参数，格式化逻辑不再感知城市/行业/时间上下文。
    """
    if not rows:
        return "暂无数据", {}

    if template_key == "company_count":
        total = rows[0].get("total", 0)
        return f"平台共有 {total} 家企业", {"total": total}

    if template_key == "company_pending":
        pending = rows[0].get("total", 0)
        return f"待审核企业共 {pending} 家", {"pending": pending}

    if template_key == "job_count":
        _status_label = {0: "未审核", 1: "已发布", 2: "不通过", 3: "停止发布"}
        parts = [
            f"{_status_label.get(r.get('status'), '未知')}（{r.get('cnt', 0)} 个）"
            for r in rows
        ]
        return "职位状态分布：" + "、".join(parts), {"rows": rows}

    if template_key == "job_pending":
        total = rows[0].get("total", 0)
        return f"待审核职位共 {total} 个", {"total": total}

    if template_key == "apply_count":
        total = rows[0].get("total", 0)
        return f"平台总报名记录共 {total} 条", {"total": total}

    return "查询完成", {"rows": rows}


# ─────────────────────────────────────────────
# 多模板并行执行与结果合并（v3：移除 params 参数）
# ─────────────────────────────────────────────

def _execute_multi_templates(keys: list[str], tenant_id: int) -> dict:
    """
    并行执行多个固定模板 SQL，合并结果。
    单条执行失败时跳过（不终止整体），仍返回其余成功结果。
    v3 变更：移除 params 参数，_build_fixed_sql 和 _format_fixed_result 签名同步简化。

    返回：
        {
            "message": "结果1；结果2",
            "stats":   {合并的结构化数据}
        }
    """
    messages: list[str] = []
    stats:    dict      = {}

    for key in keys:
        sql = _build_fixed_sql(key, tenant_id)
        if sql is None:
            logger.warning(f"[platform_stats] 模板 {key} SQL 构建失败，跳过")
            continue

        rows, err = _execute_sql(sql)
        if err:
            logger.error(f"[platform_stats] 模板 {key} 执行失败: {err}")
            continue

        msg, stat = _format_fixed_result(key, rows)
        messages.append(msg)

        # 合并 stats，分组类数据用带前缀的 key 区分，避免多模板结果互相覆盖
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


def _error_response(message: str) -> dict:
    return {
        "intent": "platform_stats",
        "data":   {"message": message},
        "status": "error",
    }


# ─────────────────────────────────────────────
# LLM 路径（支持单对象 / 数组两种返回格式）
# ─────────────────────────────────────────────

def _call_llm_raw(query: str, tenant_id: int, llm: ChatOpenAI) -> str:
    """调用 LLM 生成 SQL，返回清洗后的原始字符串（去除 ``` 围栏）"""
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
    logger.info(f"[platform_stats] 走 LLM 路径: '{query}'")

    try:
        raw    = _call_llm_raw(query, tenant_id, llm)
        parsed = json.loads(raw)
    except Exception as e:
        logger.error(f"[platform_stats] LLM SQL 解析失败: {e}")
        return _error_response("SQL 生成失败，请换个问法重试")

    # 统一为列表处理（单对象包装为单元素列表）
    items: list[dict] = parsed if isinstance(parsed, list) else [parsed]

    messages:  list[str]  = []
    rows_all:  list[dict] = []

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
    平台统计 Agent 主入口（v3 重构）。

    路由流程（优先级从高到低）：
        1. 含时间词（今天/本月/上个月等）
               → 直接走 LLM（固定 SQL 无时间条件，结果会不正确）
        2. 命中固定模板 + 含分析词（对比/趋势/分析/分布等）
               → 整句走 LLM（保留完整语义，LLM 处理分析需求）
        3. 命中固定模板，无分析词
               → 单模板：执行对应预设 SQL，返回结构化结果
               → 多模板：并行执行，合并结果
        4. 未命中任何固定模板
               → 直接走 LLM，不返回引导语（v3 移除引导语）

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

    # ── 时间词前置检测（v3 新增）─────────────────────────────────
    # 含时间词时直接走 LLM，不进入固定模板匹配，避免固定 SQL 丢失时间条件。
    # 例："上个月报名总数"会命中 apply_count，但固定 SQL 查全量，结果不正确。
    if any(sig in query for sig in _TIME_SIGNALS):
        logger.info(f"[platform_stats] 含时间词，直接走 LLM: '{query}'")
        return _handle_llm(query, tenant_id, llm)

    matched_keys = _dedup_keys(_match_fixed_templates(query))

    # ── 命中固定模板，但含分析对比词 → 整句走 LLM ────────────────
    if matched_keys and _has_analysis_signals(query):
        logger.info(f"[platform_stats] 命中模板 {matched_keys} 但含分析词，走 LLM: '{query}'")
        return _handle_llm(query, tenant_id, llm)

    # ── 命中固定模板，无分析词 → 预设 SQL 路径 ────────────────────
    if matched_keys:
        if len(matched_keys) == 1:
            # 单模板命中
            key  = matched_keys[0]
            sql  = _build_fixed_sql(key, tenant_id)
            if sql is None:
                # 理论上不会发生（5 个模板均有固定 SQL），兜底走 LLM
                return _handle_llm(query, tenant_id, llm)

            rows, err = _execute_sql(sql)
            if err:
                return _error_response("数据查询失败，请稍后重试")

            message, stats = _format_fixed_result(key, rows)
            if "rows" in stats:
                return _build_response(message, rows=stats["rows"])
            return _build_response(message, stats=stats)

        else:
            # 多模板命中：并行执行，合并结果
            logger.info(f"[platform_stats] 多模板并行执行: {matched_keys}")
            result = _execute_multi_templates(matched_keys, tenant_id)
            return _build_response(result["message"], stats=result["stats"] or None)

    # ── 未命中任何固定模板 → 直接走 LLM（v3 移除引导语兜底）────────
    return _handle_llm(query, tenant_id, llm)
