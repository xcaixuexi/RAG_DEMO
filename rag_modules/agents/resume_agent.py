"""
resume_agent.py — 简历解析 Agent

处理流程：
    文件路径
        ↓
    文件解析层（PyMuPDF / python-docx）→ 纯文本
        ↓
    按 user_role 分支：
        jobseeker → LLM 解析 JSON → 存库 → LLM 生成优化建议 → 返回前端
        recruiter → LLM 解析+分析 → 适配度摘要 → 返回前端（不存库）

函数签名：
    handle(file_path, user_role, history) -> dict
"""

import os
import json
import logging
from typing import Optional

import fitz                          # PyMuPDF，pip install pymupdf
from docx import Document            # python-docx

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 文件解析层
# ─────────────────────────────────────────────

def _extract_text(file_path: str) -> str:
    """
    根据文件后缀选择解析器，返回纯文本。
    支持 .pdf（PyMuPDF）和 .docx（python-docx）。
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    else:
        raise ValueError(f"不支持的文件类型：{ext}，请上传 .pdf 或 .docx 文件")


def _parse_pdf(file_path: str) -> str:
    """PyMuPDF 提取 PDF 文字，保留段落换行"""
    text_blocks = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_blocks.append(page.get_text("text"))
    raw = "\n".join(text_blocks).strip()
    if not raw:
        raise ValueError("PDF 文件内容为空，可能是扫描件，暂不支持 OCR 解析")
    logger.info(f"PDF 解析完成，字符数: {len(raw)}")
    return raw


def _parse_docx(file_path: str) -> str:
    """python-docx 提取 Word 文字"""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    raw = "\n".join(paragraphs).strip()
    if not raw:
        raise ValueError("Word 文件内容为空")
    logger.info(f"DOCX 解析完成，字符数: {len(raw)}")
    return raw


# ─────────────────────────────────────────────
# Prompt 定义
# ─────────────────────────────────────────────

# 求职者：解析为结构化 JSON（存库用）
_PARSE_SYSTEM = """你是一个专业的简历解析器。
从用户提供的简历文本中提取结构化信息，严格按照以下 JSON 格式输出，不要输出任何解释或多余文字：

{
    "name": "姓名",
    "age": "年龄",
    "education": "最高学历",
    "years_of_experience": "工作年限",
    "current_position": "当前/最近职位",
    "skills": [
        {"skill_name": "技能名称", "skill_level": "掌握程度"}
    ],
    "experience": [
        {
            "company": "公司名称",
            "position": "职位",
            "start_date": "开始时间",
            "end_date": "结束时间",
            "description": "工作描述"
        }
    ]
}

信息缺失时对应字段填空字符串""，技能和经历为空时填空数组[]。"""

# 求职者：基于解析结果生成优化建议（返回前端）
_JOBSEEKER_ADVICE_SYSTEM = """你是一位资深招聘顾问，专注于帮助求职者优化简历。
根据候选人的简历信息，给出具体、可执行的简历优化建议。
语气友好专业，建议条理清晰，聚焦在表达方式、信息完整度、亮点突出等方面。"""

# 招聘者：解析+分析，返回摘要（不存库）
_RECRUITER_ANALYSIS_SYSTEM = """你是一位专业的招聘顾问，协助招聘者快速评估候选人简历。
请对简历进行全面分析，输出包含以下内容的评估报告：
1. 候选人基本信息摘要（姓名、学历、工作年限、当前职位）
2. 核心技能和专业亮点
3. 职业发展轨迹分析
4. 综合评价与适用岗位建议
语言简洁专业，便于招聘者快速决策。"""


# ─────────────────────────────────────────────
# LLM 调用工具
# ─────────────────────────────────────────────

def _llm_call(llm: ChatOpenAI, system: str, user_content: str) -> str:
    """封装单次 LLM 调用"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human",  "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_content}).strip()


def _parse_json_safe(text: str) -> Optional[dict]:
    """
    安全解析 LLM 返回的 JSON，兼容带 ```json 围栏的情况。
    解析失败返回 None。
    """
    # 去除 markdown 代码块围栏
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}\n原始文本: {text[:200]}")
        return None


# ─────────────────────────────────────────────
# 业务分支
# ─────────────────────────────────────────────

def _handle_jobseeker(
    raw_text: str,
    llm: ChatOpenAI,
    user_id: int,
) -> dict:
    """
    求职者流程：
        1. LLM 解析 → JSON → 存库
        2. LLM 生成优化建议 → 返回前端
    """
    # Step 1：解析为 JSON
    parse_result = _llm_call(llm, _PARSE_SYSTEM, raw_text)
    parsed = _parse_json_safe(parse_result)

    if parsed is None:
        return _error_response("简历解析失败，请确认文件内容完整后重试")

    # Step 2：存库
    try:
        from db.mysql_client import MySQLClient
        db = MySQLClient()
        candidate_id = db.save_candidate(
            user_id  = user_id,
            parsed   = parsed,
            raw_text = raw_text,
        )
        logger.info(f"简历已存库: candidate_id={candidate_id}")
    except Exception as e:
        # 存库失败不阻断主流程，记录日志后继续
        logger.error(f"简历存库失败: {e}")
        candidate_id = None

    # Step 3：生成优化建议
    summary = (
        f"姓名：{parsed.get('name', '未知')}\n"
        f"学历：{parsed.get('education', '未知')}\n"
        f"年限：{parsed.get('years_of_experience', '未知')}\n"
        f"当前职位：{parsed.get('current_position', '未知')}\n"
        f"技能：{', '.join(s.get('skill_name','') for s in parsed.get('skills', []))}"
    )
    advice = _llm_call(llm, _JOBSEEKER_ADVICE_SYSTEM, f"以下是候选人简历摘要：\n{summary}")

    return {
        "intent": "resume_parse",
        "data": {
            "message":      advice,
            "candidate_id": candidate_id,   # 供前端后续关联使用
            "parsed":       parsed,         # 供前端展示结构化信息
        },
        "status": "success",
    }


def _handle_recruiter(raw_text: str, llm: ChatOpenAI) -> dict:
    """
    招聘者流程：
        LLM 解析 + 分析 → 适配度摘要 → 返回前端（不存库）
    """
    analysis = _llm_call(llm, _RECRUITER_ANALYSIS_SYSTEM, raw_text)

    return {
        "intent": "resume_parse",
        "data": {"message": analysis},
        "status": "success",
    }


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _error_response(message: str) -> dict:
    return {
        "intent": "resume_parse",
        "data":   {"message": message},
        "status": "error",
    }


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    file_path: str,
    user_role: str = "recruiter",
    history:   Optional[list[dict]] = None,
    llm:       Optional[ChatOpenAI] = None,
    user_id:   int = 0,
) -> dict:
    """
    简历解析 Agent 主入口。

    Args:
        file_path: 简历文件的本地路径（.pdf 或 .docx）
        user_role: "recruiter" | "jobseeker"，决定业务分支
        history:   多轮对话历史（当前暂未使用，预留接口）
        llm:       ChatOpenAI 实例，由 ChatController 从 Supervisor 传入
        user_id:   求职者账号 ID，jobseeker 分支存库时使用

    Returns:
        统一响应字典：
            jobseeker → {"intent": "resume_parse", "data": {"message": 建议, "candidate_id": id, "parsed": {...}}, "status": "success"}
            recruiter → {"intent": "resume_parse", "data": {"message": 摘要}, "status": "success"}
    """
    history = history or []

    # 校验 llm
    if llm is None:
        logger.error("[resume_agent] llm 未传入")
        return _error_response("系统配置错误，请联系管理员")

    # 文件解析
    try:
        raw_text = _extract_text(file_path)
    except ValueError as e:
        return _error_response(str(e))
    except FileNotFoundError:
        return _error_response(f"文件不存在：{file_path}")
    except Exception as e:
        logger.error(f"[resume_agent] 文件解析异常: {e}")
        return _error_response("文件解析失败，请检查文件是否损坏")

    # 按角色分支
    if user_role == "jobseeker":
        return _handle_jobseeker(raw_text, llm, user_id)
    else:
        # recruiter 及其他角色均走招聘者分析流程
        return _handle_recruiter(raw_text, llm)
