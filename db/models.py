"""
db/models.py — SQLAlchemy ORM 表结构定义

表关系：
    candidates ──< candidate_skills
    candidates ──< candidate_experience
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    ForeignKey, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    """候选人主表"""
    __tablename__ = "candidates"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    user_id             = Column(Integer, nullable=False, index=True, comment="关联系统用户ID")
    name                = Column(String(64),  nullable=False, default="")
    age                 = Column(String(16),  nullable=False, default="")   # 存字符串兼容"28岁"/"28"
    education           = Column(String(64),  nullable=False, default="")
    years_of_experience = Column(String(16),  nullable=False, default="")
    current_position    = Column(String(128), nullable=False, default="")
    raw_text            = Column(Text,        nullable=False, default="")   # 简历原始文本备份
    created_at          = Column(DateTime, nullable=False, default=datetime.utcnow)

    skills     = relationship("CandidateSkill",      back_populates="candidate", cascade="all, delete-orphan")
    experience = relationship("CandidateExperience", back_populates="candidate", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Candidate id={self.id} name={self.name}>"


class CandidateSkill(Base):
    """候选人技能表"""
    __tablename__ = "candidate_skills"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    skill_name   = Column(String(64),  nullable=False, default="")
    skill_level  = Column(String(32),  nullable=False, default="")   # 如 "熟练" / "了解" / "精通"

    candidate = relationship("Candidate", back_populates="skills")

    def __repr__(self):
        return f"<CandidateSkill {self.skill_name}({self.skill_level})>"


class CandidateExperience(Base):
    """候选人工作经历表"""
    __tablename__ = "candidate_experience"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    company      = Column(String(128), nullable=False, default="")
    position     = Column(String(128), nullable=False, default="")
    start_date   = Column(String(32),  nullable=False, default="")   # 存字符串兼容 "2020-03" / "2020年3月"
    end_date     = Column(String(32),  nullable=False, default="")   # "至今" 也可存
    description  = Column(Text,        nullable=False, default="")

    candidate = relationship("Candidate", back_populates="experience")

    def __repr__(self):
        return f"<CandidateExperience {self.company} - {self.position}>"
