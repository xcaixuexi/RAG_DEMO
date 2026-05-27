"""
unknown_agent.py — 未知意图 Agent

定位：尽力理解意图不明确的问题，能联系招聘场景则引导，实在无法理解时礼貌提示。
每次独立回答，不维护多轮历史。

API 响应统一格式：
    {"intent": "unknown", "data": {"message": "..."}, "status": "success"}
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

_SYSTEM_PROMPT = """你是一个招聘AI助手。
用户发来了一条你无法明确归类的问题，请按以下策略处理：

1. 尝试理解用户意图，给出最相关的回答
2. 如果问题能联系到招聘场景（求职、招人、简历、面试、薪酬等），主动引导到招聘话题
3. 如果实在无法理解，礼貌提示用户换个方式描述，并举例说明你能帮助的范围

语气友好，不生硬拒绝，回复简洁不冗长。"""


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _build_response(message: str, status: str = "success") -> dict:
    return {
        "intent": "unknown",
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
    兜底 Agent 主入口。

    Args:
        query:     用户输入
        user_role: 用户角色（预留，暂不用于分支）
        history:   多轮历史（不需要，保留签名一致性）
        llm:       ChatOpenAI 实例，由 ChatController 从 Supervisor 传入

    Returns:
        统一响应字典：{"intent": "unknown", "data": {"message": "..."}, "status": "success"}
    """
    if llm is None:
        logger.error("[unknown_agent] llm 未传入")
        return _build_response("系统配置错误，请联系管理员", status="error")

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human",  "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({"query": query}).strip()
        logger.info(f"[unknown_agent] 兜底回答生成完成，长度: {len(answer)}")
    except Exception as e:
        logger.error(f"[unknown_agent] LLM 调用失败: {e}")
        return _build_response("抱歉，我暂时无法理解您的问题，请换个方式描述，或告诉我您的招聘需求。", status="error")

    return _build_response(answer)
