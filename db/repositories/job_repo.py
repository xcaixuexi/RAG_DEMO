"""
db/repositories/job_repo.py — 职位与候选人查询封装

只读查询，不做任何写入操作。
所有对外方法返回 list[dict]，方便 Agent 直接序列化进响应。

安全约定：
    execute_job_query 执行前强制校验 SQL 以 SELECT 开头，
    且不含 INSERT / UPDATE / DELETE / DROP / TRUNCATE 关键词。
"""

import re
import logging
from typing import Optional

from sqlalchemy import text

from db.mysql_client import MySQLClient

logger = logging.getLogger(__name__)

# ── 允许的字段列表（LLM 生成 SQL 时的参考，repo 层不做强制列限制）
_JOB_FIELDS = (
    "id, name, company_name, company_logo, salary, salary_min, salary_max, "
    "job_exp, education, job_type, job_duty, work_city, "
    "contact_name, contact_phone, welfare"
)

# ── 禁止出现在 LLM 生成 SQL 中的关键词（大小写不敏感）
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> None:
    """
    SQL 安全校验：
        1. 必须以 SELECT 开头（忽略前置空白）
        2. 不能含有写操作关键词
    不合规时抛 ValueError，由调用方处理。
    """
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError(f"SQL 安全校验失败：只允许 SELECT 语句，收到：{stripped[:60]}")
    if _FORBIDDEN_KEYWORDS.search(stripped):
        raise ValueError(f"SQL 安全校验失败：包含禁止关键词，SQL：{stripped[:60]}")


# ── 状态中文映射 ──────────────────────────────
_STATUS_MAP = {
    1: "审核中",
    2: "未通过",
    3: "在职",
    4: "已离职",
    5: "报名取消",
}

_AUDIT_TYPE_MAP = {
    1: "待审核",
    2: "录用",
    3: "不适合",
}

_EMP_WAY_MAP = {
    0: "自主报名",
    1: "代替报名",
}


def _row_to_job_dict(row) -> dict:
    """将 SQLAlchemy Row 对象转为职位字典"""
    return {
        "id":            row.id,
        "name":          row.name          or "",
        "company_name":  row.company_name  or "",
        "company_logo":  row.company_logo  or "",
        "salary":        row.salary        or "面议",
        "salary_min":    row.salary_min,
        "salary_max":    row.salary_max,
        "work_city":     row.work_city     or "",
        "job_exp":       row.job_exp       or "不限",
        "education":     row.education     or "不限",
        "welfare":       row.welfare       or "",
        "job_type":      row.job_type,
        "job_duty":      row.job_duty      or "",
        "contact_name":  row.contact_name  or "",
        "contact_phone": row.contact_phone or "",
    }


def _row_to_apply_dict(row) -> dict:
    """将三表 JOIN 的 Row 对象转为候选人完整字典，含状态中文映射"""
    def _fmt_dt(dt):
        return dt.strftime("%Y-%m-%d %H:%M") if dt else ""

    status    = row.status
    audit_type = row.audit_type
    emp_way   = getattr(row, "emp_way", None)

    return {
        "apply_id":        row.apply_id,
        "user_id":         row.user_id,
        "resume_id":       row.resume_id,
        "job_id":          row.job_id,
        "job_name":        row.job_name      or "",
        "company_id":      row.company_id,
        "company_name":    row.company_name  or "",
        "expected_salary": float(row.expected_salary) if row.expected_salary else None,
        "status":          status,
        "status_label":    _STATUS_MAP.get(status, str(status)),
        "audit_type":      audit_type,
        "audit_type_label":_AUDIT_TYPE_MAP.get(audit_type, str(audit_type)),
        "emp_way":         emp_way,
        "emp_way_label":   _EMP_WAY_MAP.get(emp_way, "") if emp_way is not None else "",
        "create_time":     _fmt_dt(row.create_time),
        "audit_time":      _fmt_dt(row.audit_time),
        "cancel_time":     _fmt_dt(getattr(row, "cancel_time", None)),
        "remark":          getattr(row, "remark", "") or "",
        "reason":          getattr(row, "reason", "") or "",
    }


class JobRepo:
    """
    职位与候选人查询 Repository。
    每次实例化复用同一个 MySQLClient（连接池），
    生产建议在应用层做单例。
    """

    def __init__(self, db: Optional[MySQLClient] = None):
        self._db = db or MySQLClient()

    # ── 求职者：执行 LLM 生成的职位查询 SQL ──────

    def execute_job_query(self, sql: str) -> list[dict]:
        """
        执行 LLM 生成的 SELECT SQL（求职者职位查询），返回职位列表。
        """
        try:
            _validate_sql(sql)
        except ValueError as e:
            logger.error(f"[job_repo] {e}")
            return []

        try:
            with self._db._session() as session:
                rows = session.execute(text(sql)).fetchall()
                result = [_row_to_job_dict(r) for r in rows]
                logger.info(f"[job_repo] execute_job_query 返回 {len(result)} 条")
                return result
        except Exception as e:
            logger.error(f"[job_repo] SQL 执行失败: {e} | SQL: {sql[:200]}")
            return []

    def execute_apply_query(self, sql: str) -> list[dict]:
        """
        执行 LLM 生成的 SELECT SQL（招聘者候选人查询），返回候选人列表。
        与 execute_job_query 的区别：使用 _row_to_apply_dict 转换行数据。
        """
        try:
            _validate_sql(sql)
        except ValueError as e:
            logger.error(f"[job_repo] {e}")
            return []

        try:
            with self._db._session() as session:
                rows = session.execute(text(sql)).fetchall()
                result = [_row_to_apply_dict(r) for r in rows]
                logger.info(f"[job_repo] execute_apply_query 返回 {len(result)} 条")
                return result
        except Exception as e:
            logger.error(f"[job_repo] apply SQL 执行失败: {e} | SQL: {sql[:200]}")
            return []

    # ── 招聘者：按职位查询候选人 ─────────────────

    def get_candidates_by_job(self, job_id: int) -> list[dict]:
        """
        按 job_id 三表 JOIN 查询报名记录，返回候选人完整信息列表。
        过滤掉已取消报名（ea.status != 5）。
        """
        sql = f"""
            SELECT
                ea.id           AS apply_id,
                ea.user_id,
                ea.resume_id,
                ea.job_id,
                COALESCE(j.name,  ea.job_name)          AS job_name,
                ea.company_id,
                COALESCE(c.name,  ea.work_company_name)  AS company_name,
                ea.expected_salary,
                ea.status,
                ea.audit_type,
                ea.emp_way,
                ea.create_time,
                ea.audit_time,
                ea.cancel_time,
                ea.remark,
                ea.reason
            FROM employees_apply ea
            LEFT JOIN job     j ON ea.job_id     = j.id
            LEFT JOIN company c ON ea.company_id = c.id
            WHERE ea.job_id = {int(job_id)}
              AND ea.status != 5
            ORDER BY ea.create_time DESC
            LIMIT 50
        """
        try:
            with self._db._session() as session:
                rows = session.execute(text(sql)).fetchall()
                result = [_row_to_apply_dict(r) for r in rows]
                logger.info(f"[job_repo] job_id={job_id} 找到 {len(result)} 名候选人")
                return result
        except Exception as e:
            logger.error(f"[job_repo] get_candidates_by_job 失败: {e}")
            return []

    def get_candidates_by_job_name(self, job_name: str) -> list[dict]:
        """
        按职位名称模糊三表 JOIN 查询报名记录。
        同时匹配 job.name 和 ea.job_name（冗余字段），过滤已取消报名。
        """
        safe_name = job_name.replace("'", "''")   # 基础防注入
        sql = f"""
            SELECT
                ea.id           AS apply_id,
                ea.user_id,
                ea.resume_id,
                ea.job_id,
                COALESCE(j.name,  ea.job_name)          AS job_name,
                ea.company_id,
                COALESCE(c.name,  ea.work_company_name)  AS company_name,
                ea.expected_salary,
                ea.status,
                ea.audit_type,
                ea.emp_way,
                ea.create_time,
                ea.audit_time,
                ea.cancel_time,
                ea.remark,
                ea.reason
            FROM employees_apply ea
            LEFT JOIN job     j ON ea.job_id     = j.id
            LEFT JOIN company c ON ea.company_id = c.id
            WHERE (j.name LIKE '%{safe_name}%' OR ea.job_name LIKE '%{safe_name}%')
              AND ea.status != 5
            ORDER BY ea.create_time DESC
            LIMIT 50
        """
        try:
            with self._db._session() as session:
                rows = session.execute(text(sql)).fetchall()
                result = [_row_to_apply_dict(r) for r in rows]
                logger.info(f"[job_repo] job_name={job_name!r} 找到 {len(result)} 名候选人")
                return result
        except Exception as e:
            logger.error(f"[job_repo] get_candidates_by_job_name 失败: {e}")
            return []
