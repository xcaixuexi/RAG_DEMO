"""
db/mysql_client.py — MySQL 连接与查询操作封装

连接目标：192.168.110.8:3306 / dcz_ai
只读查询，不做任何写入操作，供 match_agent 等 Agent 使用。

"""

import os
import logging
from contextlib import contextmanager
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus

from db.models import Base, Company, Job, EmployeesApply

logger = logging.getLogger(__name__)


class MySQLClient:
    """
    封装 SQLAlchemy 连接池与只读查询。
    job 表的有效记录需同时满足：is_delete=0、status=1、audit_status=1。
    """

    def __init__(
        self,
        host:      Optional[str] = None,
        port:      Optional[int] = None,
        user:      Optional[str] = None,
        password:  Optional[str] = None,
        db:        Optional[str] = None,
        pool_size: int  = 5,
        echo:      bool = False,
    ):
        """
        连接参数优先使用传入值，否则读取 .env 环境变量：
            MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB

        默认值对应项目数据库，MYSQL_USER 和 MYSQL_PASSWORD 必须在 .env 中配置。
        """
        load_dotenv()
        host     = host     or os.getenv("MYSQL_HOST",     "192.168.110.8")
        port     = port     or int(os.getenv("MYSQL_PORT", "3306"))
        user     = user     or os.getenv("MYSQL_USER",     "dev_user_ai")
        password = password or os.getenv("MYSQL_PASSWORD", "@Aa123456@")
        db       = db       or os.getenv("MYSQL_DB",       "dcz_ai")

        encoded_password = quote_plus(password)
        url = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{db}?charset=utf8mb4"
        self._engine = create_engine(
            url,
            pool_size=pool_size,
            pool_pre_ping=True,
            echo=echo,
        )
        self._Session = sessionmaker(bind=self._engine)
        # 只读模式，不调用 create_all，不修改现有表结构
        logger.info(f"MySQLClient 初始化完成 → {db}@{host}:{port}")

    @contextmanager
    def _session(self):
        """只读会话上下文，不 commit"""
        session = self._Session()
        try:
            yield session
        finally:
            session.close()

    # ── 企业查询 ──────────────────────────────

    def get_company(self, company_id: int) -> Optional[Company]:
        """按 id 查询企业（apply_status=1 表示已通过审核）"""
        with self._session() as s:
            return (
                s.query(Company)
                .filter(Company.id == company_id, Company.is_delete == 0)
                .first()
            )

    def get_all_companies(self) -> list[Company]:
        """查询所有已审核通过、未删除的企业"""
        with self._session() as s:
            return (
                s.query(Company)
                .filter(Company.is_delete == 0, Company.apply_status == 1)
                .order_by(Company.create_time.desc())
                .all()
            )

    # ── 职位查询 ──────────────────────────────

    def get_job(self, job_id: int) -> Optional[Job]:
        """按 id 查询单个职位"""
        with self._session() as s:
            return (
                s.query(Job)
                .filter(Job.id == job_id, Job.is_delete == 0)
                .first()
            )

    def get_published_jobs(self) -> list[Job]:
        """
        查询所有已发布且审核通过的有效职位。
        条件：is_delete=0 AND status=1 AND audit_status=1
        """
        with self._session() as s:
            return (
                s.query(Job)
                .filter(
                    Job.is_delete    == 0,
                    Job.status       == 1,
                    Job.audit_status == 1,
                )
                .order_by(Job.deploy_time.desc())
                .all()
            )

    def get_jobs_by_company(self, company_id: int) -> list[Job]:
        """查询某企业下所有已发布职位"""
        with self._session() as s:
            return (
                s.query(Job)
                .filter(
                    Job.company_id   == company_id,
                    Job.is_delete    == 0,
                    Job.status       == 1,
                    Job.audit_status == 1,
                )
                .order_by(Job.deploy_time.desc())
                .all()
            )

    def search_jobs(self, keyword: str) -> list[Job]:
        """
        按关键词模糊搜索职位（职位名称 + 工作职责 + 职位要求）。
        供 match_agent 根据用户 JD 描述检索相关职位使用。
        """
        with self._session() as s:
            like = f"%{keyword}%"
            return (
                s.query(Job)
                .filter(
                    Job.is_delete    == 0,
                    Job.status       == 1,
                    Job.audit_status == 1,
                    or_(
                        Job.name.like(like),
                        Job.job_duty.like(like),
                        Job.job_require.like(like),
                        Job.work_kind_name.like(like),
                    ),
                )
                .order_by(Job.deploy_time.desc())
                .all()
            )

    # ── 报名记录查询 ──────────────────────────

    def get_application(self, apply_id: int) -> Optional[EmployeesApply]:
        """按 id 查询单条报名记录"""
        with self._session() as s:
            return (
                s.query(EmployeesApply)
                .filter(EmployeesApply.id == apply_id)
                .first()
            )

    def get_applications_by_job(self, job_id: int) -> list[EmployeesApply]:
        """查询某职位的所有报名记录，按报名时间倒序"""
        with self._session() as s:
            return (
                s.query(EmployeesApply)
                .filter(EmployeesApply.job_id == job_id)
                .order_by(EmployeesApply.create_time.desc())
                .all()
            )

    def get_applications_by_user(self, user_id: int) -> list[EmployeesApply]:
        """查询某用户的所有报名记录"""
        with self._session() as s:
            return (
                s.query(EmployeesApply)
                .filter(EmployeesApply.user_id == user_id)
                .order_by(EmployeesApply.create_time.desc())
                .all()
            )
