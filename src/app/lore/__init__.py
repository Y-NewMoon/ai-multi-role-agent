"""V2.1 世界书知识库（lore）：解析 / 切分 / 索引 / 检索 / 服务。

模块边界（ADR-0009）：lore 是「文档知识」，与语义记忆（MemoryStore）相互独立；
不复用 LangGraph Store，也不进记忆 namespace。
"""
