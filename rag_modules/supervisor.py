"""
rag_modules/supervisor.py — 路由主管（重构版）

主要改进：
    1. 单例模式：进程内只初始化一次 LLM，避免每次请求重建
    2. 上下文感知路由：接收上一轮 intent/query，解决指代/省略句误判
    3. 结构化 Prompt：分块编号 + 全局优先规则 + 边界案例示例 + 置信度输出
    4. 委托 LLMFactory：统一模型切换，业务层零感知

v2 变更：
    - Intent 新增 platform_stats（平台管理员专属统计意图）
    - LLM Router Prompt 新增第 8 个意图说明
    - 全局优先规则新增 P5：含跨公司聚合词时优先 platform_stats

v3（消歧补丁）变更：
    - platform_stats 意图说明末尾新增边界排除示例，防止 LLM 误判
    - 全局优先规则 P5 收紧：含公司归属词时不触发 platform_stats
"""

import logging
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.llm_factory import LLMFactory
from utils.timer import timed

logger = logging.getLogger(__name__)

# 意图类型定义
Intent = Literal[
    "resume_parse",
    "job_search",        # 求职者搜索平台公开职位
    "job_manage",        # 招聘者查看/管理自己公司职位
    "candidate_search",  # 招聘者查询候选人和报名情况
    "platform_stats",    # 平台管理员专属统计查询（新增）
    "knowledge",
    "chitchat",
    "unknown",
]

_VALID_INTENTS = {
    "resume_parse", "job_search", "job_manage",
    "candidate_search", "platform_stats", "knowledge", "chitchat",
}


# ─────────────────────────────────────────────
# 路由 Prompt（结构化重写）
# v3 变更：
#   - platform_stats 说明末尾新增【边界排除规则】
#   - 全局优先规则 P5 收紧，加入"不含公司归属词"限制
# ─────────────────────────────────────────────

_ROUTER_PROMPT_TEMPLATE = """\
你是招聘AI助手的意图分类器。将用户输入精确分类为下列意图之一。

═══════════════════════════════════════════════
当前用户角色：{user_role}
上一轮意图：{prev_intent}
上一轮问题：{prev_query}
═══════════════════════════════════════════════

【全局优先规则 — 先判断再看详细说明】
P1. 含"招/招聘/招募/招一个/帮我招/我想招" → 优先 candidate_search（招聘者视角）
P2. 含"找工作/找职位/想应聘/投简历/求职" → 优先 job_search（求职者视角）
P3. 含"简历/cv/履历" + 操作动词 → 优先 resume_parse
P4. 当前输入 ≤8字 且为省略句/指代（"深圳的呢"/"还有吗"/"那个职位"）
    → 继承上一轮意图（prev_intent），不要输出 unknown
P5. 含"所有企业/所有公司/全平台/平台统计/租户统计"等跨公司聚合词，
    且不含"我们公司/本公司/公司职位/公司发布"等公司归属词 → 优先 platform_stats

═══════════════════════════════════════════════
【意图详细说明】
═══════════════════════════════════════════════

1. resume_parse — 解析/分析简历（招聘者与求职者均可）
   触发词：简历、cv、履历、工作经历、教育背景、简历评分、简历优化
   示例：
   ✓ "帮我分析这份简历"
   ✓ "这份cv怎么样"
   ✓ "提取候选人的工作经历"
   ✗ "找简历" → candidate_search（找人不是解析简历）

2. job_search — 求职者在平台搜索公开职位
   触发词：找工作、找职位、想应聘、投简历、有哪些岗位、推荐职位
   示例：
   ✓ "深圳有没有Python开发的职位"
   ✓ "我想找一份月薪15k以上的产品经理岗位"
   ✓ "推荐一些上海的前端职位"
   ✓ "有没有临时工岗位"
   ✗ "帮我招一个前端" → candidate_search

3. job_manage — 招聘者查看/管理本公司职位
   触发词：我们公司、公司发布的、公司职位、公司岗位、管理职位
   示例：
   ✓ "我们公司现在有哪些在招职位"
   ✓ "公司发布了多少个岗位"
   ✓ "查看我们公司停止发布的职位"
   ✓ "我们公司发布了多少职位"
   ✓ "公司有多少个在招岗位"
   ✗ "查所有职位" → 角色不明确时倾向 job_search

4. candidate_search — 招聘者查询候选人/描述岗位需求找人
   触发词：招、招聘、招募、候选人、报名情况、筛选简历、找人才、帮我招
   示例：
   ✓ "帮我招一个3年经验的Java后端"
   ✓ "产品经理职位有多少人报名"
   ✓ "帮我筛选一下候选人"
   ✓ "我想招一个做运营的"
   ✓ "找一个有销售经验的人"
   ✓ "产品经理职位报名了多少人"
   ✗ "我想找一份销售工作" → job_search（我=求职者）

5. knowledge — 招聘知识/法规/流程问答
   触发词：如何、怎么、注意事项、劳动法、试用期、offer、薪酬、面试技巧、JD
   示例：
   ✓ "面试时要注意什么"
   ✓ "试用期法律规定是什么"
   ✓ "如何写招聘JD"
   ✓ "薪酬谈判有什么技巧"

6. chitchat — 日常问候/闲聊（与招聘无关）
   示例：
   ✓ "你好"
   ✓ "今天天气怎么样"
   ✓ "你叫什么名字"

7. unknown — 真正无法归类时使用（慎用，优先继承上下文）

8. platform_stats — 平台级统计查询（仅 admin 可用）
   触发词：平台、所有企业、全平台、总数、统计、分布、各城市、各行业、待审核
   示例：
   ✓ "现在有多少家企业"
   ✓ "平台今天新增了多少职位"
   ✓ "各城市职位分布情况"
   ✓ "待审核的企业有几家"
   ✓ "全平台在招职位有多少"
   ✓ "所有企业的行业分布"

   【边界排除规则】
   以下场景含有公司归属词或职位级报名词，不属于平台聚合统计，请勿归为 platform_stats：
   ✗ "我们公司有多少职位" → job_manage（含"我们公司"，是招聘者查本公司，不是全平台）
   ✗ "产品经理职位报名了多少人" → candidate_search（针对具体职位的报名查询，不是平台总数）
   ✗ "公司发布了多少岗位" → job_manage（有公司归属，非平台聚合）
   ✗ "本公司在招职位数量" → job_manage（本公司视角，非平台视角）

═══════════════════════════════════════════════
【上下文继承示例】
═══════════════════════════════════════════════
prev_intent=job_search, prev_query="深圳有Python职位吗"
current="上海的呢" → job_search（继承，换城市追问）

prev_intent=candidate_search, prev_query="产品经理有多少候选人"
current="前端的呢" → candidate_search（继承，换岗位追问）

prev_intent=knowledge, prev_query="试用期有什么规定"
current="那离职呢" → knowledge（继承，话题延伸）

═══════════════════════════════════════════════
【置信度说明】
═══════════════════════════════════════════════
输出格式：意图标签|置信度
置信度：high（规则明确命中）/ medium（推断）/ low（上下文继承或猜测）

重要：只输出"意图标签|置信度"，不输出任何其他文字。

用户问题：{query}

分类结果："""


# ─────────────────────────────────────────────
# Supervisor 单例
# ─────────────────────────────────────────────

class Supervisor:
    """
    招聘AI助手路由主管（单例版）。
    路由策略（两级漏斗）：
        Level-1  RuleRouter（规则层）  — 关键词 + 正则，零 LLM 调用，毫秒级
        Level-2  LLM Router（模型层） — query_router，处理模糊输入

    单例保证：
        同一进程内只有一个 Supervisor 实例，LLM 连接只初始化一次。
        通过 Supervisor.get_instance() 获取，不要直接实例化。

    上下文感知路由：
        route() 接受 prev_intent / prev_query 参数，
        解决"深圳的呢"/"还有吗"等省略句的意图继承问题。
    """

    _instance: Optional["Supervisor"] = None

    def __new__(cls, *args, **kwargs):
        # 单例：已存在实例时直接返回，不重新初始化
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        instance._initialized = False
        cls._instance = instance
        return instance

    def __init__(
        self,
        provider:                  Optional[str] = None,
        model_name:                Optional[str] = None,
        temperature:               float = 0.0,
        top_p:                     float = 0.9,
        max_tokens:                int   = 2048,
        api_key:                   Optional[str] = None,
        base_url:                  Optional[str] = None,
        rule_confidence_threshold: int  = 1,
        enable_rule_router:        bool = True,
    ):
        # 防止重复初始化（单例第二次调用时跳过）
        if self._initialized:
            return

        self.enable_rule_router = enable_rule_router
        self.stats = {"rule_hit": 0, "llm_hit": 0, "total": 0}

        # 通过工厂创建 LLM，支持任意供应商切换
        self.llm = LLMFactory.create(
            provider    = provider,
            model_name  = model_name,
            temperature = temperature,
            top_p       = top_p,
            max_tokens  = max_tokens,
            api_key     = api_key,
            base_url    = base_url,
        )

        if self.enable_rule_router:
            from rag_modules.rule_router import RuleRouter
            self._rule_router = RuleRouter(confidence_threshold=rule_confidence_threshold)
        else:
            self._rule_router = None

        self._initialized = True
        logger.info("[Supervisor] 初始化完成（单例）")

    @classmethod
    def get_instance(
        cls,
        provider:                  Optional[str] = None,
        model_name:                Optional[str] = None,
        temperature:               float = 0.0,
        rule_confidence_threshold: int   = 1,
        enable_rule_router:        bool  = True,
    ) -> "Supervisor":
        """
        获取单例实例的推荐方式。
        首次调用时按参数初始化；之后参数被忽略，直接返回已有实例。
        如需更换模型，先调用 reset_instance()。
        """
        if cls._instance is None:
            cls(
                provider                  = provider,
                model_name                = model_name,
                temperature               = temperature,
                rule_confidence_threshold = rule_confidence_threshold,
                enable_rule_router        = enable_rule_router,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        销毁单例，下次 get_instance() 时重新初始化。
        用于：热更新模型配置、测试隔离。
        """
        cls._instance = None
        LLMFactory.clear_cache()
        logger.info("[Supervisor] 单例已重置")

    # ==================== 对外主接口 ====================

    @timed("路由主管总耗时")
    def route(
        self,
        query:       str,
        user_role:   str = "jobseeker",
        prev_intent: Optional[str] = None,
        prev_query:  Optional[str] = None,
    ) -> tuple[str, Intent]:
        """
        两级路由主入口，供 ChatController 调用。

        Args:
            query:       用户原始输入
            user_role:   用户角色
            prev_intent: 上一轮路由结果（用于省略句继承）
            prev_query:  上一轮用户输入（提供上下文）

        Returns:
            (processed_query, intent)
        """
        self.stats["total"] += 1

        # ── Level-1：规则路由（快速路径，不感知角色/历史）──
        if self._rule_router:
            intent = self._rule_router.route(query)
            if intent is not None:
                self.stats["rule_hit"] += 1
                logger.info(
                    f"[规则路由命中] '{query}' → {intent} "
                    f"(规则命中率: {self._rule_hit_rate():.1%})"
                )
                return query, intent

        # ── Level-2：LLM 路由（感知角色 + 历史上下文）──
        # rewritten = self.query_rewrite(query)  # 可选的查询重写，暂时不启用，保持原 query 直接路由
        # intent, confidence = self._query_router(rewritten, user_role, prev_intent, prev_query)
        intent, confidence = self._query_router(query, user_role, prev_intent, prev_query)
        self.stats["llm_hit"] += 1
        logger.info(
            # f"[LLM路由命中] '{query}' → rewrite='{rewritten}' → {intent} "
            f"[LLM路由命中] '{query}' → {intent} "
            f"[{confidence}] (LLM命中率: {self._llm_hit_rate():.1%})"
        )
        # return rewritten, intent
        return query, intent

    def get_stats(self) -> dict:
        """返回当前命中率统计，供监控/日志使用"""
        total = self.stats["total"] or 1  # 防止除零
        return {
            **self.stats,
            "rule_hit_rate": round(self.stats["rule_hit"] / total, 4),
            "llm_hit_rate":  round(self.stats["llm_hit"] / total, 4),
        }

    # ==================== LLM 路由（内部）====================

    @timed("LLM路由")
    def _query_router(
        self,
        query:       str,
        user_role:   str,
        prev_intent: Optional[str] = None,
        prev_query:  Optional[str] = None,
    ) -> tuple[Intent, str]:
        """
        LLM 意图分类（带上下文感知 + 置信度输出）。

        Returns:
            (intent, confidence)  confidence ∈ {"high", "medium", "low"}
        """
        prompt = ChatPromptTemplate.from_template(_ROUTER_PROMPT_TEMPLATE)
        chain  = prompt | self.llm | StrOutputParser()

        raw = chain.invoke({
            "query":       query,
            "user_role":   user_role,
            "prev_intent": prev_intent or "无",
            "prev_query":  prev_query  or "无",
        }).strip().lower()

        # 解析 "intent|confidence" 格式
        intent, confidence = self._parse_router_output(raw, query)
        logger.info(f"[LLM路由] raw='{raw}' → intent={intent} confidence={confidence}")
        return intent, confidence

    @staticmethod
    def _parse_router_output(raw: str, query: str) -> tuple[Intent, str]:
        """
        解析 LLM 输出，支持：
            "job_search|high"
            "job_search"         (无置信度时默认 medium)
            "job_search high"    (空格分隔兼容)
        """
        # 兼容管道符和空格两种分隔
        for sep in ("|", " "):
            if sep in raw:
                parts = raw.split(sep, 1)
                intent_raw  = parts[0].strip()
                confidence  = parts[1].strip() if len(parts) > 1 else "medium"
                break
        else:
            intent_raw = raw.strip()
            confidence = "medium"

        # 清理可能的残留标点
        import re
        intent_raw = re.sub(r"[^\w_]", "", intent_raw)

        if intent_raw in _VALID_INTENTS:
            return intent_raw, confidence  # type: ignore

        logger.warning(f"[LLM路由] 无法解析意图 '{raw}'，降级为 unknown")
        return "unknown", "low"

    # ==================== 查询重写（可选启用）====================

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写，仅当 query 模糊时生效。
        目前 Supervisor.route() 默认不调用，可在需要时手动启用。
        """
        prompt = PromptTemplate(
            template="""\
你是招聘领域查询分析助手。判断用户查询是否需要重写以提高处理效果。

原始查询: {query}

分析规则：
1. **直接返回原查询（无需重写）的情况**：
   - 已经包含明确实体或指令：如"解析这份简历"、"推荐python开发候选人"、"面试注意事项"
   - 清晰的操作请求：如"匹配产品经理岗位"、"上传简历文件"
   - 具体的知识问题：如"招聘法务需要哪些资质"

2. **需要重写的情况**：
   - 过于模糊或宽泛：如"找人"、"看简历"、"面一下"
   - 缺少关键信息：如"开发"、"销售"、"实习生"
   - 口语化、省略主语：如"有没有合适的"、"帮我看看"
   - 需要补全为完整的招聘场景表述

重写原则：
- 保持原意，补全缺失的关键词（岗位名称、操作类型等）
- 统一使用"解析简历"、"匹配岗位"、"查询知识"等清晰动词
- 不添加原文没有的意图

示例：
- "找人" → "匹配候选人"
- "开发" → "推荐软件开发岗位候选人"
- "面一下" → "面试相关问题"
- "有没有合适的" → "匹配岗位推荐"
- "解析这份简历" → "解析这份简历"（保持不变）
- "python开发需要什么技能" → "python开发需要什么技能"（保持不变）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"],
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    # ==================== 私有辅助 ====================

    def _rule_hit_rate(self) -> float:
        return self.stats["rule_hit"] / (self.stats["total"] or 1)

    def _llm_hit_rate(self) -> float:
        return self.stats["llm_hit"] / (self.stats["total"] or 1)