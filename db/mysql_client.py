"""
db/mysql_client.py — MySQL 连接池（单例版）

改进：
    - 单例模式：进程内只创建一个 SQLAlchemy Engine，避免每次请求重建连接池
    - get_instance() 作为推荐入口，__init__ 保留向后兼容
    - 连接池参数可通过环境变量或 get_instance() 参数覆盖
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker

from db.models import Base, Company, Job, EmployeesApply

logger = logging.getLogger(__name__)


class MySQLClient:
    """
    封装 SQLAlchemy 连接池与只读查询（单例）。
    job 表的有效记录需同时满足：is_delete=0、status=1、audit_status=1。
    """

    _instance: Optional["MySQLClient"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        instance._initialized = False
        cls._instance = instance
        return instance

    def __init__(
        self,
        host:       Optional[str] = None,
        port:       Optional[int] = None,
        user:       Optional[str] = None,
        password:   Optional[str] = None,
        db:         Optional[str] = None,
        pool_size:  int  = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,   # 1小时后回收空闲连接，防止 MySQL 8h 超时断开
        echo:       bool = False,
    ):
        if self._initialized:
            return

        """
        连接参数优先使用传入值，否则读取 .env 环境变量：
            MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB

        默认值对应项目数据库，MYSQL_USER 和 MYSQL_PASSWORD 必须在 .env 中配置。
        """
        load_dotenv()
        host      = host     or os.getenv("MYSQL_HOST",     "192.168.110.8")
        port      = port     or int(os.getenv("MYSQL_PORT", "3306"))
        user      = user     or os.getenv("MYSQL_USER",     "dev_user_ai")
        password  = password or os.getenv("MYSQL_PASSWORD", "@Aa123456@")
        db        = db       or os.getenv("MYSQL_DB",       "dcz_ai")

        encoded_password = quote_plus(password)
        url = (
            f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{db}"
            f"?charset=utf8mb4"
        )
        self._engine = create_engine(
            url,
            pool_size      = pool_size,
            max_overflow   = max_overflow,
            pool_recycle   = pool_recycle,
            pool_pre_ping  = True,   # 每次获取连接前 ping，自动重连
            echo           = echo,
        )
        self._Session    = sessionmaker(bind=self._engine)
        self._initialized = True
        logger.info(
            f"[MySQLClient] 连接池初始化完成 → {db}@{host}:{port} "
            f"(pool_size={pool_size}, max_overflow={max_overflow})"
        )

    @classmethod
    def get_instance(cls, **kwargs) -> "MySQLClient":
        """
        获取单例的推荐方式。
        首次调用时按参数初始化，之后参数被忽略。
        """
        if cls._instance is None:
            cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """销毁单例（测试隔离用）"""
        if cls._instance and hasattr(cls._instance, "_engine"):
            cls._instance._engine.dispose()
        cls._instance = None
        logger.info("[MySQLClient] 单例已重置")

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
        """按 id 查询企业"""
        with self._session() as s:
            return (
                s.query(Company)
                .filter(Company.id == company_id, Company.is_delete == 0)
                .first()
            )

    def get_company_by_user_id(self, user_id: int) -> Optional[Company]:
        """
        根据申请人 user_id 查询企业。
        需满足 is_delete=0 且 apply_status=1（已通过审核）。
        供控制器层获取招聘者关联的 company_id 使用。
        """
        with self._session() as s:
            return (
                s.query(Company)
                .filter(
                    Company.user_id      == user_id,
                    Company.is_delete    == 0,
                    Company.apply_status == 1,
                )
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
        供 job_search_agent 根据用户 JD 描述检索相关职位使用。
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
