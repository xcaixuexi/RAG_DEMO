"""
test_rule_router.py — RuleRouter 单元测试（无需 LLM，离线可运行）

运行方式：
    python -m pytest test_rule_router.py -v
    # 或直接执行
    python test_rule_router.py
"""

import sys
import os

# 让测试可以直接在项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 临时 mock rag_modules 路径（测试时 rule_router 单独运行）
# 实际项目中删除此段，直接 from rag_modules.rule_router import RuleRouter
import types

# 构造最小 mock，使 Intent 类型可导入
mock_supervisor = types.ModuleType("rag_modules.supervisor")
mock_supervisor.Intent = str  # type: ignore
sys.modules.setdefault("rag_modules", types.ModuleType("rag_modules"))
sys.modules["rag_modules.supervisor"] = mock_supervisor

from rule_router import RuleRouter  # noqa: E402

router = RuleRouter(confidence_threshold=1)


# ─────────────────────────────────────────────
# 测试用例（query, expected_intent）
# expected_intent=None 表示应放行给 LLM
# ─────────────────────────────────────────────

CASES: list[tuple[str, str | None]] = [
    # ── resume_parse ──────────────────────────
    ("帮我解析这份简历",            "resume_parse"),
    ("这份cv怎么样",                "resume_parse"),
    ("简历分析",                    "resume_parse"),
    ("提取候选人工作经历",          "resume_parse"),
    ("分析简历优劣",                "resume_parse"),
    ("请看看这份履历",              "resume_parse"),
    ("给我简历评分",                "resume_parse"),

    # ── job_match ─────────────────────────────
    ("招聘python开发，推荐候选人",  "job_match"),
    ("这个JD匹配哪些简历",          "job_match"),
    ("帮我找个产品经理",            "job_match"),
    ("推荐人才",                    "job_match"),
    ("找候选人",                    "job_match"),
    ("帮我筛选简历",                "job_match"),
    ("我想招一个前端工程师",        "job_match"),

    # ── knowledge ─────────────────────────────
    ("面试时要注意什么",            "knowledge"),
    ("如何写招聘JD",                "knowledge"),
    ("试用期法律要求是什么",        "knowledge"),
    ("劳动合同有哪些规定",          "knowledge"),
    ("薪酬谈判技巧",                "knowledge"),
    ("背调需要注意什么",            "knowledge"),
    ("什么是猎头",                  "knowledge"),
    ("offer怎么写",                 "knowledge"),

    # ── chitchat ──────────────────────────────
    ("你好",                        "chitchat"),
    ("hello",                       "chitchat"),
    ("谢谢",                        "chitchat"),
    ("再见",                        "chitchat"),
    ("你是谁",                      "chitchat"),
    ("今天天气怎么样",              "chitchat"),
    ("你有什么功能",                "chitchat"),

    # ── 否定保护 → 放行给 LLM ─────────────────
    ("不要帮我解析简历",            None),
    ("我不需要推荐候选人",          None),

    # ── 模糊输入 → 放行给 LLM ────────────────
    ("找人",                        None),   # 太短，规则不命中（由 LLM rewrite）
    ("开发",                        None),   # 单词，无上下文
]


def run_tests():
    passed = 0
    failed = 0
    skipped = 0

    print(f"\n{'─'*60}")
    print(f"{'输入':<28} {'期望':<16} {'实际':<16} {'结果'}")
    print(f"{'─'*60}")

    for query, expected in CASES:
        actual = router.route(query)

        if expected is None:
            # 期望放行（None），实际也是 None 则通过
            ok = actual is None
        else:
            ok = actual == expected

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"{query:<28} {str(expected):<16} {str(actual):<16} {status}"
        )

    print(f"{'─'*60}")
    print(f"共 {len(CASES)} 条 | 通过 {passed} | 失败 {failed}\n")

    return failed == 0


def run_explain_demo():
    """展示 explain() 的调试输出"""
    demo_queries = [
        "帮我找个产品经理",
        "今天股票涨了吗",
        "不需要解析简历",
        "开发",
    ]
    print("\n── explain() 调试输出 ────────────────────────")
    for q in demo_queries:
        info = router.explain(q)
        print(f"\n查询: {q!r}")
        print(f"  归一化: {info['normalized']!r}")
        print(f"  否定保护: {info['negation_guard']}")
        print(f"  得分: {info['scores']}")
        print(f"  → 结果: {info['result']}")


if __name__ == "__main__":
    success = run_tests()
    run_explain_demo()
    sys.exit(0 if success else 1)
