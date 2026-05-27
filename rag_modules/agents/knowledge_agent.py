"""
knowledge_agent.py — 招聘知识问答 Agent

定位：资深招聘顾问"小才"，以招聘领域为主，其他问题简短配合回答。
每次独立回答，不维护多轮历史。

API 响应统一格式：
    {"intent": "knowledge", "data": {"message": "..."}, "status": "success"}
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一位资深招聘顾问AI助手。

回答规则：
1. 招聘相关问题（面试技巧、劳动法规、薪酬标准、招聘流程、JD撰写、背调、offer等）：
   - 深入解答，分点列举，条理清晰
   - 涉及法律法规时给出参考依据（如《劳动合同法》第X条），并提示"以实际政策为准"
2. 非招聘问题：
   - 简短回答，不超过3句话，之后引导回招聘场景
3. 语气专业、严肃，避免口语化表达
4. 回答结构清晰，适当使用序号或分点"""


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _build_response(message: str, status: str = "success") -> dict:
    return {
        "intent": "knowledge",
        "data":   {"message": message},
        "status": status,
    }


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query:     str,
    user_role: str = "recruiter",
    history:   Optional[list[dict]] = None,
    llm:       Optional[ChatOpenAI] = None,
) -> dict:
    """
    招聘知识问答 Agent 主入口。

    Args:
        query:     用户问题
        user_role: 用户角色（预留，暂不用于分支）
        history:   多轮历史（不需要，保留签名一致性）
        llm:       ChatOpenAI 实例，由 ChatController 从 Supervisor 传入

    Returns:
        统一响应字典：{"intent": "knowledge", "data": {"message": "..."}, "status": "success"}
    """
    if llm is None:
        logger.error("[knowledge_agent] llm 未传入")
        return _build_response("系统配置错误，请联系管理员", status="error")

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human",  "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({"query": query}).strip()
        logger.info(f"[knowledge_agent] 回答生成完成，长度: {len(answer)}")
    except Exception as e:
        logger.error(f"[knowledge_agent] LLM 调用失败: {e}")
        return _build_response("抱歉，暂时无法回答，请稍后重试。", status="error")

    return _build_response(answer)
