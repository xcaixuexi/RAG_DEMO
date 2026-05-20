"""
rule_router.py — 基于关键词/正则的规则快速路由层

设计目标：
    在调用 LLM 之前，对高频、意图明确的用户输入直接命中路由意图，
    完全跳过 query_rewrite + query_router 两次 LLM 调用，节省 Token 资源。

路由优先级（从高到低）：
    1. 否定保护 (NegationGuard)  — 含"不要/别/取消"等否定词时直接放行给 LLM
    2. resume_parse              — 简历解析类
    3. job_match                 — 岗位匹配/人才推荐类
    4. knowledge                 — 招聘知识/流程/法规类
    5. chitchat                  — 闲聊/问候类
    返回 None 表示规则未命中，交由 LLM 处理。

使用方式：
    from rag_modules.rule_router import RuleRouter
    router = RuleRouter()
    intent = router.route("帮我解析这份简历")   # → "resume_parse" 或 None
"""

import re
import logging
from typing import Optional
from rag_modules.supervisor import Intent  # 复用已有类型定义

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """统一全角→半角、去除多余空格，便于正则命中"""
    text = text.strip()
    # 全角转半角（数字、字母）
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif ch == "\u3000":
            result.append(" ")
        else:
            result.append(ch)
    return "".join(result)


def _any_keyword(text: str, keywords: list[str]) -> bool:
    """文本中是否包含 keywords 列表中任意一个关键词"""
    return any(kw in text for kw in keywords)


def _match_any_pattern(text: str, patterns: list[re.Pattern]) -> bool:
    """文本是否匹配 patterns 列表中任意一个正则"""
    return any(p.search(text) for p in patterns)


# ─────────────────────────────────────────────
# 规则数据集（按意图分组，便于维护扩展）
# ─────────────────────────────────────────────

# 1. 否定保护关键词：含这些词时，意图模糊，交给 LLM
_NEGATION_WORDS: list[str] = [
    "不要", "别", "取消", "停止", "不用", "不需要", "不想", "不是",
    "没有", "没", "并非", "而非",
]

# ── resume_parse ──────────────────────────────
_RESUME_KEYWORDS: list[str] = [
    "简历", "履历", "cv", "resume",
    "求职者", "候选人信息", "应聘者",
    "工作经历", "项目经历", "教育背景", "学历", "技能栈",
    "解析", "提取信息", "分析简历", "看简历", "读简历",
    "简历评分", "简历优化", "简历建议",
]

_RESUME_PATTERNS: list[re.Pattern] = [
    re.compile(r"(帮我|请|麻烦).{0,6}(解析|分析|看看|提取|评估).{0,6}简历"),
    re.compile(r"这份(简历|cv|履历)"),
    re.compile(r"简历.{0,10}(怎么样|如何|优劣|分析|打分|评价)"),
    re.compile(r"(提取|抽取|识别).{0,8}(技能|经验|学历|工作经历)"),
    re.compile(r"(上传|发送|给你).{0,6}简历"),
]

# ── job_match ─────────────────────────────────
_MATCH_KEYWORDS: list[str] = [
    "匹配候选人", "推荐候选人", "推荐人才", "找候选人", "找人才",
    "筛选简历",   # 注意："筛选简历"归 job_match，纯"简历"归 resume_parse
    "招聘需求", "岗位需求", "职位描述", "招募", "招人",
    "合适的人", "符合条件", "人才库", "候选人推荐",
    "寻找人才",
    # 招聘动词 + 职位词组合（不单独用"jd"，避免误判知识类）
    "招前端", "招后端", "招开发", "招运营", "招设计", "招产品", "招销售",
    "我想招", "帮我招",
]

_MATCH_PATTERNS: list[re.Pattern] = [
    re.compile(r"招聘.{0,15}(推荐|匹配|找|筛选)"),
    re.compile(r"(找|推荐|匹配).{0,10}(工程师|开发|产品经理|运营|设计师|销售|hr|测试|数据|算法|前端|后端|全栈)"),
    # "这个JD匹配/找/推荐" — job_match
    re.compile(r"(这个|该).{0,4}jd.{0,10}(匹配|找|推荐|适合)"),
    re.compile(r"(哪些|哪个).{0,6}(候选人|求职者).{0,6}(符合|适合|匹配)"),
    re.compile(r"(按|根据).{0,10}(jd|职位|岗位).{0,6}(推荐|匹配|筛选)"),
    # 帮我找/招 + 职位词（不含纯"帮我找"，太模糊）
    re.compile(r"帮我(找|招|推荐).{0,10}(工程师|开发|经理|专员|设计师|运营|销售|前端|后端|测试|算法)"),
    re.compile(r"(想招|要招|需要招).{0,15}(工程师|开发|经理|专员|设计师|运营|销售|前端|后端|测试)"),
    # "筛选简历"是 job_match 场景，通过正则优先覆盖
    re.compile(r"(筛选|过滤).{0,6}简历"),
    # "JD 匹配简历"场景
    re.compile(r"jd.{0,10}(匹配|找|对应|适合).{0,6}简历"),
    re.compile(r"简历.{0,10}(匹配|符合|适合).{0,6}(jd|岗位|职位|需求)"),
]

# ── knowledge ─────────────────────────────────
_KNOWLEDGE_KEYWORDS: list[str] = [
    # 操作动词（限招聘场景，不用"如何/怎么"裸词，避免误伤 chitchat）
    "注意事项", "技巧", "方法", "流程", "步骤",
    # 招聘领域实体
    "试用期", "劳动合同", "劳动法", "薪酬", "薪资结构", "背调", "背景调查",
    "offer模板", "入职流程", "离职流程", "绩效考核", "kpi考核",
    "面试题", "面试官", "面试流程", "面试技巧",
    "招聘流程", "猎头", "人力资源", "hr知识",
    "什么是", "解释一下", "介绍一下",
    "写jd", "撰写jd", "制作jd",   # 知识类"写JD"场景
]

_KNOWLEDGE_PATTERNS: list[re.Pattern] = [
    # "如何/怎么 + 招聘场景动词" — 精确限定，不裸用"怎么"
    re.compile(r"(如何|怎么|怎样).{0,20}(面试|招聘|谈薪|入职|离职|背调|考核|写jd|制作jd)"),
    re.compile(r"(面试|招聘|试用期|劳动法|薪酬|offer).{0,15}(注意|技巧|要求|规定|标准|流程|步骤)"),
    re.compile(r"(什么是|介绍|解释).{0,15}(背调|猎头|offer|kpi|绩效|试用期|劳动合同)"),
    # 写/撰写 JD 是知识类
    re.compile(r"(写|撰写|制作|怎么写).{0,6}(jd|职位描述|招聘要求|offer)"),
    re.compile(r"(offer|jd).{0,6}(怎么写|如何写|模板|范本|格式)"),
    re.compile(r"招聘.{0,10}(需要注意|流程|步骤|标准|规范)"),
    re.compile(r"(劳动|合同|法律).{0,10}(规定|要求|条款|风险)"),
]

# ── chitchat ──────────────────────────────────
_CHITCHAT_KEYWORDS: list[str] = [
    "你好", "hello", "hi", "嗨", "哈喽",
    "早上好", "下午好", "晚上好", "早安", "晚安",
    "谢谢", "感谢", "thanks", "thank you",
    "再见", "拜拜", "bye", "下次见",
    "你是谁", "你叫什么", "你是什么", "你能做什么", "你有什么功能",
    # 非招聘话题（独立实体词，不依赖"怎么"）
    "天气", "股票", "新闻", "吃什么", "心情",
    "股价", "行情", "涨了", "跌了", "今日指数",
    "哈哈", "笑死", "有意思", "好玩",
]

_CHITCHAT_PATTERNS: list[re.Pattern] = [
    # 纯问候（短句锚定首尾）
    re.compile(r"^(你好|hello|hi|嗨|哈喽)[！!。.？?～~\s]*$", re.IGNORECASE),
    re.compile(r"^(早上好|下午好|晚上好|早安|晚安)[！!。.？?～~\s]*$"),
    re.compile(r"^(谢谢|感谢|thanks|thank\s*you)[！!。.，,\s]*$", re.IGNORECASE),
    re.compile(r"^(再见|拜拜|bye)[！!。.？?\s]*$", re.IGNORECASE),
    # 关于 AI 身份的闲聊
    re.compile(r"你(是谁|叫什么名字|是什么模型|能做什么|有什么功能)[？?\s]*$"),
    # 与招聘完全无关的话题 — 含"天气/股票/行情"的任意句子
    re.compile(r"(今天|明天|最近|现在|当前).{0,6}(天气|股票|行情|股价|指数|涨跌)"),
    re.compile(r"(天气|股票|股价|行情).{0,10}(怎么样|如何|好不好|涨了|跌了)"),
]

# ── 高置信度短句映射（完全匹配，优先于关键词） ──
# key: 精确短语  value: 意图
_EXACT_MAP: dict[str, Intent] = {
    # resume_parse
    "解析简历": "resume_parse",
    "分析简历": "resume_parse",
    "看简历": "resume_parse",
    "提取简历信息": "resume_parse",
    "简历分析": "resume_parse",
    "简历解析": "resume_parse",
    # job_match
    "匹配候选人": "job_match",
    "推荐候选人": "job_match",
    "推荐人才": "job_match",
    "找候选人": "job_match",
    "筛选简历": "job_match",
    "招聘匹配": "job_match",
    # chitchat
    "你好": "chitchat",
    "hi": "chitchat",
    "hello": "chitchat",
    "嗨": "chitchat",
    "谢谢": "chitchat",
    "感谢": "chitchat",
    "再见": "chitchat",
    "拜拜": "chitchat",
}


# ─────────────────────────────────────────────
# 核心路由类
# ─────────────────────────────────────────────

class RuleRouter:
    """
    基于关键词 + 正则的轻量规则路由器。

    命中时：直接返回 Intent（跳过 LLM）。
    未命中时：返回 None，由 Supervisor 继续 LLM 流程。

    置信度机制：
        每个意图通过"关键词命中数 + 正则命中数"累加得分，
        只有得分 ≥ confidence_threshold 才认为命中，防止单词误判。
    """

    def __init__(self, confidence_threshold: int = 1):
        """
        Args:
            confidence_threshold: 最低命中分数（默认 1）。
                精确短句映射直接命中，不受此阈值约束。
                关键词和正则各贡献 1 分，可调高阈值提升精确率。
        """
        self.confidence_threshold = confidence_threshold
        logger.info(
            f"RuleRouter 初始化完成，置信度阈值={confidence_threshold}"
        )

    # ── 公开接口 ──────────────────────────────

    def route(self, query: str) -> Optional[Intent]:
        """
        尝试对 query 进行规则路由。

        Args:
            query: 原始用户输入（route 内部会自动归一化）

        Returns:
            命中的 Intent 字符串，或 None（交由 LLM 处理）
        """
        text = _normalize(query).lower()

        # 0. 精确短句映射（最高优先级，零歧义）
        intent = self._exact_match(text)
        if intent:
            logger.info(f"[规则路由] 精确命中 '{query}' → {intent}")
            return intent

        # 1. 否定保护：含否定词时放行给 LLM，避免误判
        if _any_keyword(text, _NEGATION_WORDS):
            logger.debug(f"[规则路由] 含否定词，放行给 LLM: '{query}'")
            return None

        # 2. 冲突裁决：job_match 正则命中时，直接优先于关键词积分（防止"简历"词干扰）
        if _match_any_pattern(text, _MATCH_PATTERNS):
            # 排除 chitchat（极低可能同时命中，保险起见检查一下）
            if not _match_any_pattern(text, _CHITCHAT_PATTERNS):
                logger.info(f"[规则路由] job_match 正则优先命中 '{query}' → job_match")
                return "job_match"

        # 3. 按意图顺序打分，返回首个得分达标的意图
        for intent_label, kw_list, pat_list in [
            ("resume_parse", _RESUME_KEYWORDS, _RESUME_PATTERNS),
            ("job_match",    _MATCH_KEYWORDS,  _MATCH_PATTERNS),
            ("knowledge",    _KNOWLEDGE_KEYWORDS, _KNOWLEDGE_PATTERNS),
            ("chitchat",     _CHITCHAT_KEYWORDS,  _CHITCHAT_PATTERNS),
        ]:
            score = self._score(text, kw_list, pat_list)
            if score >= self.confidence_threshold:
                logger.info(
                    f"[规则路由] 命中 '{query}' → {intent_label} (score={score})"
                )
                return intent_label  # type: ignore

        logger.debug(f"[规则路由] 未命中，交由 LLM: '{query}'")
        return None

    def explain(self, query: str) -> dict:
        """
        调试用：返回各意图的得分明细，帮助调整关键词/正则。

        Returns:
            {
                "normalized": "...",
                "negation_guard": True/False,
                "scores": {
                    "resume_parse": 2,
                    "job_match": 0,
                    ...
                },
                "result": "resume_parse" | None
            }
        """
        text = _normalize(query).lower()
        has_negation = _any_keyword(text, _NEGATION_WORDS)
        scores = {}
        for label, kw_list, pat_list in [
            ("resume_parse", _RESUME_KEYWORDS, _RESUME_PATTERNS),
            ("job_match",    _MATCH_KEYWORDS,  _MATCH_PATTERNS),
            ("knowledge",    _KNOWLEDGE_KEYWORDS, _KNOWLEDGE_PATTERNS),
            ("chitchat",     _CHITCHAT_KEYWORDS,  _CHITCHAT_PATTERNS),
        ]:
            scores[label] = self._score(text, kw_list, pat_list)

        result = self.route(query)
        return {
            "normalized": text,
            "negation_guard": has_negation,
            "scores": scores,
            "result": result,
        }

    # ── 私有方法 ──────────────────────────────

    def _exact_match(self, text: str) -> Optional[Intent]:
        """精确短句映射，text 已归一化小写"""
        # 去除常见标点后再匹配
        clean = re.sub(r"[！!？?。.，,～~\s]+", "", text)
        return _EXACT_MAP.get(clean) or _EXACT_MAP.get(text)

    @staticmethod
    def _score(text: str, keywords: list[str], patterns: list[re.Pattern]) -> int:
        """
        计算文本对某个意图的匹配得分：
            关键词命中一个 +1，正则命中一个 +1（上限各 3，防止关键词堆叠失真）
        """
        kw_score = min(sum(1 for kw in keywords if kw in text), 3)
        pat_score = min(sum(1 for p in patterns if p.search(text)), 3)
        return kw_score + pat_score
