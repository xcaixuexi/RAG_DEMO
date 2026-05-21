"""
chitchat_agent.py — 闲聊 Agent

两级处理结构：
    Level-1  模板匹配（无需 LLM）— 高频问候/告别/感谢/自我介绍，毫秒级响应
    Level-2  LLM 兜底            — 人设为严肃专业的招聘助手"小才"，
                                   非招聘话题配合回答但不计入对话上下文

API 响应统一格式：
    {"intent": "chitchat", "data": {"message": "..."}, "status": "success"}
"""

import re
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

_BOT_NAME = "小才"

_SELF_INTRO = (
    f"我是{_BOT_NAME}，您的专属招聘助手 🤝\n"
    "我可以帮您：\n"
    "  · 解析和分析候选人简历\n"
    "  · 根据岗位需求匹配合适人才\n"
    "  · 解答招聘流程、劳动法规、面试技巧等知识\n"
    "有什么招聘方面的问题，随时告诉我！"
)

# ── 模板映射表 ────────────────────────────────
# (正则pattern, 回复文本)
# 按优先级从高到低排列，首个命中即返回
_TEMPLATE_RULES: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"^(你好|hello|hi|嗨|哈喽|早上好|下午好|晚上好|早安|晚安)[！!。.？?～~\s]*$", re.IGNORECASE),
        f"你好！我是{_BOT_NAME}，您的专属招聘助手，有什么可以帮您？",
    ),
    (
        re.compile(r"^(再见|拜拜|bye|下次见|回头见)[！!。.？?～~\s]*$", re.IGNORECASE),
        "再见！有需要随时找我～",
    ),
    (
        re.compile(r"^(谢谢|感谢|thanks|thank\s*you|多谢)[！!。.，,\s]*$", re.IGNORECASE),
        "不客气！有其他问题欢迎继续问我 😊",
    ),
    (
        re.compile(r"(你是谁|你叫什么|你是什么|你能做什么|你有什么功能|介绍.{0,4}自己|自我介绍)[？?]?$"),
        _SELF_INTRO,
    ),
]

# ── LLM 系统提示 ──────────────────────────────
_SYSTEM_PROMPT = f"""你是{_BOT_NAME}，一个严肃专业的招聘AI助手。

人设规则：
1. 始终保持招聘助手的专业身份，语气友好但不随意
2. 招聘相关问题（简历、岗位、面试、劳动法等）优先、深入解答
3. 非招聘话题（天气、股票、娱乐等）可以简短配合回答，但主动引导回招聘场景
4. 不扮演其他角色，不回答违法违规内容
5. 回复简洁，避免冗长，中文为主

示例风格：
- 用户问天气 → 简短回应 + "顺便问一下，最近有招聘需求吗？"
- 用户闲聊心情 → 表示理解 + 引导到工作/招聘话题
"""


# ─────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────

def _template_match(query: str) -> Optional[str]:
    """模板匹配层，命中返回固定回复，未命中返回 None"""
    text = query.strip()
    for pattern, reply in _TEMPLATE_RULES:
        if pattern.search(text):
            logger.info(f"[chitchat] 模板命中: '{query}'")
            return reply
    return None


def _build_response(message: str) -> dict:
    """构造统一 API 响应格式"""
    return {
        "intent": "chitchat",
        "data": {"message": message},
        "status": "success",
    }


# ─────────────────────────────────────────────
# 对外接口
# ─────────────────────────────────────────────

def handle(
    query: str,
    history: Optional[list[dict]] = None,
    llm: Optional[ChatOpenAI] = None,
) -> dict:
    """
    闲聊 Agent 主入口。

    Args:
        query:   用户当前输入
        history: 多轮对话历史，格式为 [{"role": "user"|"assistant", "content": "..."}]
                 由 ChatController 统一管理，最多传入最近 3 轮（6 条消息）
        llm:     ChatOpenAI 实例，由 ChatController 从 Supervisor 传入，避免重复初始化

    Returns:
        统一响应字典：{"intent": "chitchat", "data": {"message": "..."}, "status": "success"}
    """
    history = history or []

    # ── Level-1：模板匹配 ─────────────────────
    reply = _template_match(query)
    if reply:
        return _build_response(reply)

    # ── Level-2：LLM 兜底 ─────────────────────
    if llm is None:
        # 降级兜底：不应走到这里，Controller 应始终传入 llm
        logger.warning("[chitchat] llm 未传入，返回默认回复")
        return _build_response("您好，我是招聘助手小才，请问有什么招聘方面的问题？")

    # 拼装消息列表：system + history + 当前问题
    # 注意：非招聘的闲聊历史不计入上下文（由 Controller 的 chitchat_history 单独管理）
    messages = [("system", _SYSTEM_PROMPT)]
    for turn in history:
        messages.append((turn["role"], turn["content"]))
    messages.append(("human", query))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm | StrOutputParser()

    try:
        reply = chain.invoke({}).strip()
        logger.info(f"[chitchat] LLM 回复: '{reply[:50]}...'")
    except Exception as e:
        logger.error(f"[chitchat] LLM 调用失败: {e}")
        reply = "抱歉，我暂时无法回答，请稍后再试。"

    return _build_response(reply)
