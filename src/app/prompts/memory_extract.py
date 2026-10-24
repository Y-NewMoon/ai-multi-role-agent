"""V2.3 事实抽取 prompt（契约 §2.3 / §8.1）。

硬要求：
- 只输出一个 JSON 对象，不要 Markdown 围栏、不要解释；
- 没有明确信息就留空，不要推测（不编造）；
- subject 只允许「用户」/「角色」；
- profile 只认 4 个白名单槽位。
"""

EXTRACT_SYSTEM_PROMPT = (
    "你是角色扮演对话的记忆抽取器。只输出一个 JSON 对象，不要输出 Markdown 围栏、"
    "不要输出解释文字、不要输出第二个 JSON。"
)

EXTRACT_PROMPT = """请从下面这一轮对话中抽取可以长期记住的事实与用户档案。

对话：
{conversation}

输出格式（严格 JSON）：
{{"facts":[{{"subject":"用户","predicate":"喜欢的茶","object":"普洱","importance":4}}],
 "profile":{{"称呼":"阿林"}}}}

规则：
1. facts 只记对话里**明确出现**的信息；没有明确信息就输出 {{"facts":[],"profile":{{}}}}，不要推测。
2. subject 只能是「用户」或「角色」；predicate 不超过 20 字；object 不超过 40 字。
3. importance 为 1~5 的整数（5 = 称呼/禁忌/重要关系这类长期关键信息）。
4. 单次最多 5 条 facts；按 importance 从高到低排列。
5. profile 只允许 4 个键：称呼 / 禁忌 / 关系阶段 / 重要日期；没有就省略该键。
6. object 为空、为「不知道」「没说」「无」等占位词时，不要输出该条。"""
