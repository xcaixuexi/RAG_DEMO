"""
db/mysql_client.py — MySQL 连接与存储操作封装

使用方式：
    from db.mysql_client import MySQLClient
    client = MySQLClient()
    candidate_id = client.save_candidate(user_id=1, parsed=parsed_dict, raw_text="...")
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from db.models import Base, Candidate, CandidateSkill, CandidateExperience

logger = logging.getLogger(__name__)


class MySQLClient:
    """
    封装 SQLAlchemy 连接池和 CRUD 操作。
    单例模式在应用层维护，此处每次实例化独立创建连接池（生产建议做成单例）。
    """

    def __init__(
        self,
        host:     Optional[str] = None,
        port:     Optional[int] = None,
        user:     Optional[str] = None,
        password: Optional[str] = None,
        db:       Optional[str] = None,
        pool_size: int = 5,
        echo:      bool = False,        # True 时打印所有 SQL，调试用
    ):
        """
        优先使用传入参数，否则读取环境变量：
            MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB
        """
        load_dotenv()
        host     = host     or os.getenv("MYSQL_HOST",     "localhost")
        port     = port     or int(os.getenv("MYSQL_PORT", "3306"))
        user     = user     or os.getenv("MYSQL_USER",     "root")
        password = password or os.getenv("MYSQL_PASSWORD", "")
        db       = db       or os.getenv("MYSQL_DB",       "recruitment")

        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"
        self._engine = create_engine(url, pool_size=pool_size, pool_pre_ping=True, echo=echo)
        self._Session = sessionmaker(bind=self._engine)

        # 首次启动自动建表（生产环境建议用 Alembic 迁移）
        Base.metadata.create_all(self._engine)
        logger.info(f"MySQLClient 初始化完成，数据库: {db}@{host}:{port}")

    # ── 上下文管理器，自动提交/回滚 ──────────────

    @contextmanager
    def _session(self) -> Session:
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── 核心写入接口 ──────────────────────────────

    def save_candidate(
        self,
        user_id:  int,
        parsed:   dict,
        raw_text: str,
    ) -> int:
        """
        将 LLM 解析结果写入三张表，返回 candidate_id。

        Args:
            user_id:  系统用户 ID（求职者账号 ID）
            parsed:   LLM 返回的结构化 JSON dict
            raw_text: 简历原始文本（备份用）

        Returns:
            新插入的 candidate.id
        """
        with self._session() as session:
            # 1. 主表
            candidate = Candidate(
                user_id             = user_id,
                name                = parsed.get("name", ""),
                age                 = str(parsed.get("age", "")),
                education           = parsed.get("education", ""),
                years_of_experience = str(parsed.get("years_of_experience", "")),
                current_position    = parsed.get("current_position", ""),
                raw_text            = raw_text,
            )
            session.add(candidate)
            session.flush()   # 获取自增 id，事务还未提交

            # 2. 技能表
            for skill in parsed.get("skills", []):
                session.add(CandidateSkill(
                    candidate_id = candidate.id,
                    skill_name   = skill.get("skill_name", ""),
                    skill_level  = skill.get("skill_level", ""),
                ))

            # 3. 经历表
            for exp in parsed.get("experience", []):
                session.add(CandidateExperience(
                    candidate_id = candidate.id,
                    company      = exp.get("company", ""),
                    position     = exp.get("position", ""),
                    start_date   = exp.get("start_date", ""),
                    end_date     = exp.get("end_date", ""),
                    description  = exp.get("description", ""),
                ))

            logger.info(f"候选人写入成功: candidate_id={candidate.id}, user_id={user_id}")
            return candidate.id

    # ── 查询接口（预留，供后续 job_match 使用） ──

    def get_candidate(self, candidate_id: int) -> Optional[Candidate]:
        """按 candidate_id 查询候选人（含关联技能和经历）"""
        with self._session() as session:
            return (
                session.query(Candidate)
                .filter(Candidate.id == candidate_id)
                .first()
            )

    def get_candidates_by_user(self, user_id: int) -> list[Candidate]:
        """查询某用户的所有简历"""
        with self._session() as session:
            return (
                session.query(Candidate)
                .filter(Candidate.user_id == user_id)
                .order_by(Candidate.created_at.desc())
                .all()
            )
