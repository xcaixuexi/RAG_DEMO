"""
rule_router.py — 基于关键词/正则的规则快速路由层

路由优先级（从高到低）：
    1. 否定保护 (NegationGuard)  — 含"不要/别/取消"等否定词时直接放行给 LLM
    2. resume_parse              — 简历解析类
    3. candidate_search          — 招聘者查询候选人（原 job_match 招聘侧）
    4. job_search                — 求职者搜索公开职位（原 job_match 求职侧）
    5. platform_stats            — 平台管理员统计查询（v2 调整：移至 job_manage 前，防止"发布了多少职位"被抢走）
    6. job_manage                — 招聘者管理自己公司职位
    7. knowledge                 — 招聘知识/流程/法规类
    8. chitchat                  — 闲聊/问候类
    返回 None 表示规则未命中，交由 LLM 处理。

注意：规则层不感知 user_role，角色权限校验由控制器层统一处理。
"""

import re
import logging
from typing import Optional
from rag_modules.supervisor import Intent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """统一全角→半角、去除多余空格"""
    text = text.strip()
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
# 规则数据集
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
    "工作经历", "项目经历", "教育背景", "技能栈",
    "解析", "提取信息", "分析简历", "看简历", "读简历",
    "简历评分", "简历优化", "简历建议",
]

_RESUME_PATTERNS: list[re.Pattern] = [
    re.compile(r"(帮我|请|麻烦).{0,6}(解析|分析|看看|提取|评估).{0,6}简历"),
    re.compile(r"这份(简历|cv|履历)"),
    re.compile(r"简历.{0,10}(怎么样|如何|优劣|分析|打分|评价)"),
    re.compile(r"(提取|抽取|识别).{0,8}(技能|经验|工作经历)"),
    re.compile(r"(上传|发送|给你).{0,6}简历"),
]

# ── job_search（求职者搜索公开职位）──────────
_JOB_SEARCH_KEYWORDS: list[str] = [
    "找工作", "找职位", "找岗位", "求职", "应聘", "投简历",
    "有没有职位", "推荐职位", "招聘信息", "在招",
    "招前端", "招后端", "招开发", "招运营", "招设计", "招产品", "招销售",
    "我想找", "想应聘", "帮我找工作",
]

_JOB_SEARCH_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(找|推荐|有没有).{0,10}"
        r"(工程师|开发|产品经理|运营|设计师|销售|测试|数据|算法|前端|后端|全栈)"
        r".{0,6}(职位|岗位|工作)"
    ),
    re.compile(r"帮我找.{0,10}(工程师|开发|经理|专员|设计师|运营|销售|前端|后端|测试|算法)"),
    re.compile(r"(想找|要找|想应聘).{0,15}(工作|职位|岗位)"),
    re.compile(r"(深圳|上海|北京|广州|东莞|杭州|成都|武汉|西安|南京).{0,10}(职位|岗位|工作)"),
    re.compile(r"(全职|兼职|实习|临时工).{0,6}(职位|岗位|工作|推荐)"),
]

# ── job_manage（招聘者管理自己公司职位）───────
_JOB_MANAGE_KEYWORDS: list[str] = [
    "我们公司", "公司职位", "公司岗位", "公司在招", "发布的职位",
    "发布的岗位", "我发布", "我们发布", "公司有多少职位",
    "查看职位", "管理职位", "职位列表",
]

_JOB_MANAGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(我们|我|本)公司.{0,10}(职位|岗位|在招|发布)"),
    re.compile(r"(查看|查询|看看).{0,6}(公司|我们).{0,6}(职位|岗位)"),
    re.compile(r"公司.{0,6}(有多少|有哪些|在招).{0,6}(职位|岗位|工作)"),
    re.compile(r"(发布|上线).{0,6}(职位|岗位).{0,6}(列表|情况|状态)"),
]

# ── candidate_search（招聘者查询候选人）───────
_CANDIDATE_SEARCH_KEYWORDS: list[str] = [
    # 查询已有数据类
    "候选人", "报名情况", "报名人员", "应聘者", "找候选人",
    "筛选简历", "匹配候选人", "推荐人才", "查报名", "有多少人报名",
    "审核候选人", "录用", "不适合",
    # 描述招聘需求类
    "帮我招", "我想招", "招募", "招一个", "招聘一个",
    "需要招", "急招", "招人", "找人才",
]

_CANDIDATE_SEARCH_PATTERNS: list[re.Pattern] = [
    # 查询已有数据类
    re.compile(r"(查看|查询|有多少).{0,10}(候选人|报名|应聘者)"),
    re.compile(r"(筛选|过滤|审核).{0,6}(简历|候选人)"),
    re.compile(r"(这个|该).{0,4}(职位|岗位|jd).{0,10}(候选人|报名|应聘)"),
    re.compile(r"(哪些|哪个).{0,6}(候选人|求职者).{0,6}(符合|适合|匹配)"),
    re.compile(r"帮我(找|推荐|筛选).{0,10}(候选人|人才|简历)"),
    # 描述招聘需求类
    re.compile(r"(帮我|想|需要|急).{0,4}招.{0,15}(工程师|经理|专员|设计师|运营|销售|开发|测试|产品|前端|后端|算法|数据)"),
    re.compile(r"招(聘|募).{0,4}(一个|一名|若干).{0,20}(经验|学历|要求|岗位|职位)"),
    re.compile(r"(找|寻).{0,4}(一个|一名).{0,20}(工程师|经理|专员|设计师|运营|销售|开发|测试|产品)"),
    re.compile(r"(岗位|职位).{0,10}(要求|条件|经验|学历).{0,10}(推荐|匹配|有没有)"),
]

# ── platform_stats（平台管理员统计查询）───────
_PLATFORM_STATS_KEYWORDS: list[str] = [
    "平台", "租户", "所有企业", "所有公司", "全平台",
    "企业总数", "公司总数", "职位总数", "岗位总数", "报名总数",
    "统计", "数据概览", "分布情况", "待审核企业", "待审核职位",
    "各城市", "各行业", "新增企业", "新增职位",
    # v2 补充：追问 / 泛问句场景（防止被 job_manage 关键词抢走）
    "发布了多少", "多少个职位", "多少个在招", "发布的职位",
    "多少家", "共有多少", "一共多少",
]

_PLATFORM_STATS_PATTERNS: list[re.Pattern] = [
    re.compile(r"(平台|租户).{0,10}(有多少|总数|统计|概览)"),
    re.compile(r"(有多少家|多少个).{0,6}(企业|公司|职位|岗位)"),
    re.compile(r"(企业|公司|职位|报名).{0,6}(分布|统计|总数|数量)"),
    re.compile(r"(各城市|各行业|按城市|按行业).{0,10}(职位|企业|统计)"),
    re.compile(r"(待审核|未审核).{0,6}(企业|职位|候选人)"),
    re.compile(r"(今天|本周|本月|最近).{0,10}(新增|注册|发布|报名)"),
]

# ── knowledge ─────────────────────────────────
_KNOWLEDGE_KEYWORDS: list[str] = [
    "注意事项", "技巧", "方法", "流程", "步骤",
    "试用期", "劳动合同", "劳动法", "薪酬", "薪资结构", "背调", "背景调查",
    "offer模板", "入职流程", "离职流程", "绩效考核", "kpi考核",
    "面试题", "面试官", "面试流程", "面试技巧",
    "招聘流程", "猎头", "人力资源", "hr知识",
    "什么是", "解释一下", "介绍一下",
    "写jd", "撰写jd", "制作jd",
]

_KNOWLEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(如何|怎么|怎样).{0,20}(面试|招聘|谈薪|入职|离职|背调|考核|写jd|制作jd)"),
    re.compile(r"(面试|招聘|试用期|劳动法|薪酬|offer).{0,15}(注意|技巧|要求|规定|标准|流程|步骤)"),
    re.compile(r"(什么是|介绍|解释).{0,15}(背调|猎头|offer|kpi|绩效|试用期|劳动合同)"),
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
    "天气", "股票", "新闻", "吃什么", "心情",
    "股价", "行情", "涨了", "跌了", "今日指数",
    "哈哈", "笑死", "有意思", "好玩",
]

_CHITCHAT_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(你好|hello|hi|嗨|哈喽)[！!。.？?～~\s]*$", re.IGNORECASE),
    re.compile(r"^(早上好|下午好|晚上好|早安|晚安)[！!。.？?～~\s]*$"),
    re.compile(r"^(谢谢|感谢|thanks|thank\s*you)[！!。.，,\s]*$", re.IGNORECASE),
    re.compile(r"^(再见|拜拜|bye)[！!。.？?\s]*$", re.IGNORECASE),
    re.compile(r"你(是谁|叫什么名字|是什么模型|能做什么|有什么功能)[？?\s]*$"),
    re.compile(r"(今天|明天|最近|现在|当前).{0,6}(天气|股票|行情|股价|指数|涨跌)"),
    re.compile(r"(天气|股票|股价|行情).{0,10}(怎么样|如何|好不好|涨了|跌了)"),
]

# ── 精确短句映射（最高优先级）──────────────────
_EXACT_MAP: dict[str, Intent] = {
    # resume_parse
    "解析简历":     "resume_parse",
    "分析简历":     "resume_parse",
    "看简历":       "resume_parse",
    "提取简历信息": "resume_parse",
    "简历分析":     "resume_parse",
    "简历解析":     "resume_parse",
    # job_search
    "找工作":       "job_search",
    "找职位":       "job_search",
    "推荐岗位":     "job_search",
    # job_manage
    "公司职位":     "job_manage",
    "我们公司职位": "job_manage",
    "查看职位":     "job_manage",
    "职位列表":     "job_manage",
    # candidate_search
    "匹配候选人":   "candidate_search",
    "推荐人才":     "candidate_search",
    "找候选人":     "candidate_search",
    "筛选简历":     "candidate_search",
    "招聘匹配":     "candidate_search",
    # platform_stats
    "企业总数":     "platform_stats",
    "职位总数":     "platform_stats",
    "平台统计":     "platform_stats",
    "报名总数":     "platform_stats",
    "数据概览":     "platform_stats",
    # chitchat
    "你好":   "chitchat",
    "hi":     "chitchat",
    "hello":  "chitchat",
    "嗨":     "chitchat",
    "谢谢":   "chitchat",
    "感谢":   "chitchat",
    "再见":   "chitchat",
    "拜拜":   "chitchat",
}


# ─────────────────────────────────────────────
# 核心路由类
# ─────────────────────────────────────────────

class RuleRouter:
    """
    基于关键词 + 正则的轻量规则路由器。
    不感知 user_role，角色权限校验统一由控制器层处理。
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
        logger.info(f"RuleRouter 初始化完成，置信度阈值={confidence_threshold}")
        # ── 公开接口 ──────────────────────────────

    def route(self, query: str) -> Optional[Intent]:
        """
        尝试对 query 进行规则路由。
        Args:
            query: 原始用户输入

        Returns:
            命中的 Intent 字符串，或 None（交由 LLM 处理）
        """
        text = _normalize(query).lower()

        # 0. 精确短句映射
        intent = self._exact_match(text)
        if intent:
            logger.info(f"[规则路由] 精确命中 '{query}' → {intent}")
            return intent

        # 1. 否定保护：含否定词时放行给 LLM，避免误判
        if _any_keyword(text, _NEGATION_WORDS):
            logger.debug(f"[规则路由] 含否定词，放行给 LLM: '{query}'")
            return None

        # 2. candidate_search 正则优先（防止"筛选简历"被 resume_parse 关键词抢走）
        if _match_any_pattern(text, _CANDIDATE_SEARCH_PATTERNS):
            if not _match_any_pattern(text, _CHITCHAT_PATTERNS):
                logger.info(f"[规则路由] candidate_search 正则优先命中 '{query}'")
                return "candidate_search"

        # 3. 按意图顺序打分
        # 打分顺序：resume_parse → candidate_search → job_search
        #           → platform_stats → job_manage → knowledge → chitchat
        # v2 调整：platform_stats 移到 job_manage 前，
        #          防止"发布了多少职位"被 job_manage 的"发布的职位"关键词抢走
        for intent_label, kw_list, pat_list in [
            ("platform_stats",   _PLATFORM_STATS_KEYWORDS,   _PLATFORM_STATS_PATTERNS),    # v2 调整：移至 job_manage 前
            ("resume_parse",     _RESUME_KEYWORDS,           _RESUME_PATTERNS),
            ("candidate_search", _CANDIDATE_SEARCH_KEYWORDS, _CANDIDATE_SEARCH_PATTERNS),  # 提前，避免"招人"类被 job_search 抢走
            ("job_search",       _JOB_SEARCH_KEYWORDS,       _JOB_SEARCH_PATTERNS),
            ("job_manage",       _JOB_MANAGE_KEYWORDS,       _JOB_MANAGE_PATTERNS),
            ("knowledge",        _KNOWLEDGE_KEYWORDS,        _KNOWLEDGE_PATTERNS),
            ("chitchat",         _CHITCHAT_KEYWORDS,         _CHITCHAT_PATTERNS),
        ]:
            score = self._score(text, kw_list, pat_list)
            if score >= self.confidence_threshold:
                logger.info(f"[规则路由] 命中 '{query}' → {intent_label} (score={score})")
                return intent_label  # type: ignore

        logger.debug(f"[规则路由] 未命中，交由 LLM: '{query}'")
        return None

    def explain(self, query: str) -> dict:
        """调试用：返回各意图的得分明细"""
        text = _normalize(query).lower()
        has_negation = _any_keyword(text, _NEGATION_WORDS)
        scores = {}
        # explain() 的打分顺序与 route() 保持一致
        for label, kw_list, pat_list in [
            ("platform_stats",   _PLATFORM_STATS_KEYWORDS,   _PLATFORM_STATS_PATTERNS),    # v2 调整：移至 job_manage 前
            ("resume_parse",     _RESUME_KEYWORDS,           _RESUME_PATTERNS),
            ("candidate_search", _CANDIDATE_SEARCH_KEYWORDS, _CANDIDATE_SEARCH_PATTERNS),
            ("job_search",       _JOB_SEARCH_KEYWORDS,       _JOB_SEARCH_PATTERNS),
            ("job_manage",       _JOB_MANAGE_KEYWORDS,       _JOB_MANAGE_PATTERNS),
            ("knowledge",        _KNOWLEDGE_KEYWORDS,        _KNOWLEDGE_PATTERNS),
            ("chitchat",         _CHITCHAT_KEYWORDS,         _CHITCHAT_PATTERNS),
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
        kw_score  = min(sum(1 for kw in keywords if kw in text), 3)
        pat_score = min(sum(1 for p in patterns if p.search(text)), 3)
        return kw_score + pat_score
