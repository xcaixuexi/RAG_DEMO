"""
resume_agent.py — 简历解析 Agent

处理流程：
    文件路径
        ↓
    文件解析层（PyMuPDF / python-docx）→ 纯文本
        ↓
    按 user_role 分支：
        jobseeker → LLM 分析 → 优化建议返回前端
        recruiter → LLM 分析 → 适配度摘要返回前端

    两个分支均不写库。

函数签名：
    handle(file_path, user_role, history, llm, user_id) -> dict
"""

import os
import logging
from typing import Optional

import fitz                 # PyMuPDF
from docx import Document   # python-docx

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 文件解析层
# ─────────────────────────────────────────────

def _extract_text(file_path: str) -> str:
    """根据后缀选择解析器，返回纯文本"""
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
        raise ValueError("PDF 内容为空，可能是扫描件，暂不支持 OCR 解析")
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

# 求职者：生成简历优化建议
_JOBSEEKER_ADVICE_SYSTEM = """你是一位资深招聘顾问，专注于帮助求职者优化简历。
根据候选人提供的简历全文，给出具体、可执行的优化建议。
语气友好专业，建议条理清晰，聚焦在以下方面：
- 内容完整度（缺少哪些关键信息）
- 亮点突出（如何更好地展示核心竞争力）
- 表达方式（量化成果、动词选用等）
- 格式建议（如适用）"""

# 招聘者：生成候选人评估报告
_RECRUITER_ANALYSIS_SYSTEM = """你是一位专业的招聘顾问，协助招聘者快速评估候选人简历。
请对简历进行全面分析，输出结构清晰的评估报告，包含：
1. 基本信息摘要（姓名、学历、工作年限、当前职位）
2. 核心技能与专业亮点
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


# ─────────────────────────────────────────────
# 业务分支
# ─────────────────────────────────────────────

def _handle_jobseeker(raw_text: str, llm: ChatOpenAI) -> dict:
    """
    求职者流程：LLM 分析简历 → 返回优化建议
    不存库。
    """
    advice = _llm_call(llm, _JOBSEEKER_ADVICE_SYSTEM, raw_text)
    return {
        "intent": "resume_parse",
        "data":   {"message": advice},
        "status": "success",
    }


def _handle_recruiter(raw_text: str, llm: ChatOpenAI) -> dict:
    """
    招聘者流程：LLM 分析简历 → 返回评估报告
    不存库。
    """
    analysis = _llm_call(llm, _RECRUITER_ANALYSIS_SYSTEM, raw_text)
    return {
        "intent": "resume_parse",
        "data":   {"message": analysis},
        "status": "success",
    }


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
    user_role: str,
    history:   Optional[list[dict]] = None,
    llm:       Optional[ChatOpenAI] = None,
    user_id:   int = 0,              # 保留字段，当前不用于存库，供后续业务扩展
) -> dict:
    """
    简历解析 Agent 主入口。

    Args:
        file_path: 简历文件本地路径（.pdf 或 .docx）
        user_role: "recruiter" | "jobseeker"，决定返回内容风格
        history:   多轮对话历史（预留，暂未使用）
        llm:       ChatOpenAI 实例，由 ChatController 从 Supervisor 传入
        user_id:   用户 ID（保留字段，当前不使用）

    Returns:
        统一响应字典：
            {"intent": "resume_parse", "data": {"message": "..."}, "status": "success"/"error"}
    """
    history = history or []

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
        return _handle_jobseeker(raw_text, llm)
    else:
        return _handle_recruiter(raw_text, llm)
