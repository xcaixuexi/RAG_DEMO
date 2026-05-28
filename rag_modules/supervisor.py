import os
import logging
from typing import Literal, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# 定义意图类型
Intent = Literal["resume_parse", "job_match", "knowledge", "chitchat", "unknown"]


class Supervisor:
    """
    招聘AI助手路由主管 - 负责 LLM 集成、查询重写、意图路由。

    路由策略（两级漏斗）：
        Level-1  RuleRouter（规则层）  — 关键词 + 正则，零 LLM 调用，毫秒级
        Level-2  LLM Router（模型层） — query_rewrite → query_router，处理模糊输入

    命中率统计：
        通过 get_stats() 可查看两级各自的命中次数，用于后期规则调优。
    """

    def __init__(
        self,
        model_name: str = "glm-4.5-air",
        # model_name: str = "deepseek-v4-flash",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        # base_url: str = "https://api.deepseek.com",
        rule_confidence_threshold: int = 1,
        enable_rule_router: bool = True,
    ):
        """
        Args:
            model_name: 模型名称，默认 glm-4.5-air
                        切换 GPT 时传入 "gpt-4o-mini" 并将 base_url 留空或改为官方地址
            temperature: 生成温度，控制输出随机性
            top_p: 核采样参数
            max_tokens: 最大输出 Token 数
            api_key: API Key（可选，默认读取环境变量 ZHIPU_API_KEY 或 OPENAI_API_KEY）
            base_url: API 基础 URL；切换到 OpenAI 官方时传 "https://api.openai.com/v1/" 即可
            rule_confidence_threshold: 规则路由置信度阈值（≥1 即命中）
            enable_rule_router: 是否启用规则路由层（False 可退回纯 LLM 路由，便于 A/B 对比）
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_rule_router = enable_rule_router

        # 命中率统计
        self.stats = {"rule_hit": 0, "llm_hit": 0, "total": 0}

        self.llm = self._setup_llm(api_key, base_url)

        # 延迟导入，避免循环依赖
        if self.enable_rule_router:
            from rag_modules.rule_router import RuleRouter
            self._rule_router = RuleRouter(confidence_threshold=rule_confidence_threshold)
        else:
            self._rule_router = None

    def _setup_llm(self, api_key: Optional[str], base_url: str) -> ChatOpenAI:
        """初始化 ChatOpenAI，支持智谱 / OpenAI 官方 / 其他兼容接口"""
        logger.info(f"正在初始化路由主管 LLM: {self.model_name}, base_url: {base_url}")

        key = api_key
        if not key:
            load_dotenv()
            key = os.getenv("ZHIPU_API_KEY")
            # key = os.getenv("OPENAI_API_KEY")

        if not key:
            raise ValueError("请设置 ZHIPU_API_KEY 或 OPENAI_API_KEY 环境变量，或通过 api_key 参数传入")

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            api_key=key,
            base_url=base_url,
        )
        logger.info("路由主管 LLM 初始化完成")
        return llm

    # ==================== 对外主接口 ====================

    def route(self, query: str) -> tuple[str, Intent]:
        """
        两级路由主入口，供 ChatController 调用。

        Args:
            query: 用户原始输入

        Returns:
            (processed_query, intent)
            - processed_query: 规则路由时为原 query；LLM 路由时为重写后的 query
            - intent: 最终意图标签
        """
        self.stats["total"] += 1

        # ── Level-1：规则路由 ──────────────────────────
        if self._rule_router:
            intent = self._rule_router.route(query)
            if intent is not None:
                self.stats["rule_hit"] += 1
                logger.info(
                    f"[规则路由命中] '{query}' → {intent} "
                    f"(规则命中率: {self._rule_hit_rate():.1%})"
                )
                return query, intent

        # ── Level-2：LLM 路由 ─────────────────────────
        # rewritten = self.query_rewrite(query)
        # intent = self.query_router(rewritten)
        intent = self.query_router(query)
        self.stats["llm_hit"] += 1
        logger.info(
            # f"[LLM路由命中] '{query}' → rewrite='{rewritten}' → {intent} "
            f"[LLM路由命中] '{query}' → {intent} "
            f"(LLM命中率: {self._llm_hit_rate():.1%})"
        )
        # return rewritten, intent
        return query, intent

    def get_stats(self) -> dict:
        """返回当前命中率统计，供监控/日志使用"""
        total = self.stats["total"] or 1  # 防止除零
        return {
            **self.stats,
            "rule_hit_rate": round(self.stats["rule_hit"] / total, 4),
            "llm_hit_rate": round(self.stats["llm_hit"] / total, 4),
        }

    # ==================== 1. 智能查询重写 ====================

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写招聘相关的查询

        Args:
            query: 用户原始查询

        Returns:
            重写后的查询（或原查询）
        """
        prompt = PromptTemplate(
            template="""
你是一个招聘领域的智能查询分析助手。请分析用户的查询，判断是否需要重写以提高后续处理效果（如简历解析、岗位匹配、知识检索等）。

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

    # ==================== 2. 查询路由（意图识别） ====================

    def query_router(self, query: str) -> Intent:
        """
        查询路由 - 根据用户问题分类到预定义意图

        Args:
            query: 用户查询

        Returns:
            意图标签 (resume_parse, job_match, knowledge, chitchat, unknown)
        """
        prompt = ChatPromptTemplate.from_template("""
你是一个招聘AI助手的路由分类器。根据用户的问题，将其分类为以下意图之一，**只输出标签，不要输出任何其他文字**：

- **resume_parse** : 用户要求解析简历内容、提取信息（工作经历、技能、教育背景）、分析简历优劣
  示例："帮我分析这份简历"、"提取简历中的技能"、"这份简历适合什么岗位"

- **job_match** : 用户要求根据岗位描述匹配岗位，查询符合条件的职位列表，或者询问岗位相关的统计信息
  示例："目前临时工岗位有多少"、"深圳地区的职位推荐"、"帮我看看度才子公司有多少岗位在招人"

- **knowledge** : 用户询问招聘流程、面试技巧、劳动法规、薪酬标准、行业知识等
  示例："面试时要注意什么"、"如何写招聘JD"、"试用期法律要求"

- **chitchat** : 用户进行日常问候、闲聊、与工作无关的对话
  示例："你好呀"、"今天天气怎么样"、"你叫什么名字"、"今天股票行情"

- **unknown** : 无法明确归类的其他问题，或者不属于以上任何一类

用户问题: {query}

分类结果（仅标签）:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        valid_intents = ["resume_parse", "job_match", "knowledge", "chitchat"]
        if result in valid_intents:
            logger.info(f"路由决策: '{query}' → {result}")
            return result  # type: ignore
        else:
            logger.warning(f"未知路由结果 '{result}'，降级为 unknown")
            return "unknown"

    # ==================== 私有辅助 ====================

    def _rule_hit_rate(self) -> float:
        total = self.stats["total"] or 1
        return self.stats["rule_hit"] / total

    def _llm_hit_rate(self) -> float:
        total = self.stats["total"] or 1
        return self.stats["llm_hit"] / total
