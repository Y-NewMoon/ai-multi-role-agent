"""应用配置：从 .env 读取（pydantic-settings）。"""

import logging
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== 数据库 =====
    database_url: str
    test_database_url: str = ""

    # ===== LLM 对话 / 摘要 =====
    llm_provider: str = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # ===== Embedding（M2 起使用，维度定死 1024，见方案 3.6）=====
    embedding_provider: str = "local"
    embedding_model: str = "bge-m3"
    embedding_dims: int = 1024

    # ===== Prompt 预算与摘要（方案 3.9 / 3.10，M2 起使用）=====
    prompt_token_budget: int = 8000
    prompt_recent_rounds: int = 10
    # V2.3（D9）：默认 10 与 prompt_recent_rounds 对齐，消除第 11–20 轮上下文空窗
    summary_trigger_rounds: int = 10
    summary_chunk_rounds: int = 10

    # ===== 一次回复的消息条数上限（M5：角色可连发 1~N 条）=====
    reply_max_messages: int = 3
    # 内心独白补写：整轮漏写时补一次（多一次完整 prompt 的 LLM 调用，成本敏感可关）
    reply_think_retry_enabled: bool = True

    # ===== 群聊（M4，ADR-0004 §3.4/§3.7）=====
    group_max_members: int = 8  # 群内角色数上限（不含用户）
    group_max_turns: int = 10  # 一轮内最多发言次数（硬上限）

    # ===== 自主闲聊（M6：用户点击按钮触发一次）=====
    idle_group_max_messages: int = 10  # 群聊一次闲聊累计消息上限
    idle_group_turn_max_messages: int = 2  # 群聊闲聊时每个角色每次最多发言条数
    # 私聊闲聊复用 reply_max_messages（1~N 条）

    # ===== 头像上传（M5）=====
    avatar_dir: str = "data/avatars"  # 本地存储目录（相对项目根）
    avatar_url_prefix: str = "/static/avatars"  # 静态服务挂载前缀
    avatar_max_bytes: int = 2 * 1024 * 1024  # 单文件上限 2MB

    # ===== 世界书知识库（V2.1 · ADR-0009）=====
    lore_enabled: bool = True  # 关闭后 `retrieve_lore` 直接返回空（不访问数据库）
    lore_max_ratio: float = 0.20  # 检索 lore 段预算占比（契约 4.7）
    lore_always_on_ratio: float = 0.08  # always-on 段预算占比（契约 4.7）
    lore_query_rewrite: bool = True  # 查询改写开关（CI 与单测用 false）
    lore_top_k: int = 20  # 单路召回与融合后进入 rerank 的条数

    # ===== Skill 工具层（V2.2 · §2.10 / §2.5-3 / §2.4）=====
    # 静态 lore 注入模式：full（V2.1 全量）/ always_on（只注入 always-on）/ off（不注入）。
    # 供 agentic RAG 三策略对比；默认 full 保持 V2.1 行为。
    lore_static_mode: Literal["full", "always_on", "off"] = "full"
    # web_search 后端；缺 URL/KEY 时工具返回 `unavailable`（评测用 fake provider）。
    web_search_url: str = ""
    web_search_api_key: str = ""
    # 图内工具循环上限（§2.4；达到上限后强制文本收尾）。
    skill_max_tool_rounds: int = 3

    # ===== 记忆升级（V2.3 · plan §8）=====
    memory_extract_enabled: bool = True  # 关闭 = 行为退化为 V2.2
    memory_extract_min_chars: int = 8  # 用户文本短于此跳过抽取
    memory_extract_timeout_s: float = 8.0
    memory_fact_mode: Literal["overwrite", "keep_both_with_note"] = "overwrite"
    memory_fact_limit: int = 200  # 每 agent 事实条数硬上限
    memory_profile_enabled: bool = True
    memory_profile_ratio: float = 0.10  # profile 段预算占比
    memory_weighted_recall: bool = True  # False = 纯余弦（消融）
    memory_w_cosine: float = 0.6
    memory_w_recency: float = 0.25
    memory_w_importance: float = 0.15
    memory_recency_halflife_days: float = 14.0
    memory_recall_overscan: int = 50
    memory_l2_enabled: bool = True
    memory_l2_ratio: float = 0.05  # L2 段预算占比
    memory_l2_every_merges: int = 3  # 每 N 次 L1 合并重合成一次 L2（≈N×10 轮）
    memory_l2_max_chars: int = 300
    memory_recall_limit: int = 5  # 语义记忆召回条数（原写死在 retrieve_memory）

    # ===== 可观测（V2.4 · §2.11）=====
    trace_enabled: bool = True
    trace_dir: str = "data/traces"
    trace_sample_rate: float = 1.0  # 0~1
    trace_max_field_chars: int = 200
    trace_memory_stats: bool = False  # C-03 观察：true 时每轮多一次 asearch(limit=1000)
    prompt_version: str = "v3"
    # V3 M1：冻结时钟（仅评测/测试用；未设置走真实时间，见 app/core/clock.py）
    eval_fixed_now: str | None = None

    # ===== 成本（V2.4；默认仅示例，按账单覆盖）=====
    llm_price_input_cny_per_1m: float = 2.0
    llm_price_output_cny_per_1m: float = 8.0

    # ===== 稳定性（V2.4）=====
    llm_max_concurrency: int = 8
    llm_timeout_s: float = 30.0
    llm_timeout_fallback: str = "（我这边有点走神，稍等一下再说。）"
    # R1：psycopg 连接池（checkpointer + store 共用）；默认 4 太小，20 并发 SSE 会在
    # 取连接处排队（压测 p95 首 token 1.99s → 1.26s）
    db_pool_min_size: int = 4
    db_pool_max_size: int = 20

    # ===== 提醒（V2.4 · B-03）=====
    reminder_pending_ttl_hours: int = 24

    # ===== 非流式幂等（V2.4 · D8）=====
    chat_idempotency_ttl_s: int = 600
    chat_idempotency_max_entries: int = 1024

    # ===== 本机记忆接口（V3 M2）=====
    memory_api_agents: str = "ext"  # 逗号分隔白名单；纯数字 agent 一律拒绝
    memory_api_buffer_path: str = "data/memory_api/buffer.json"
    memory_api_buffer_max_messages: int = 200
    memory_api_idempotency_ttl_s: int = 600
    memory_api_idempotency_max_entries: int = 512
    memory_api_max_text_chars: int = 8000
    memory_api_max_replies: int = 20

    # ===== 压测（V2.4 · 仅 fake）=====
    fake_llm_delay_ms: int = 10
    fake_llm_reply: str = "（压测假回复）"

    @model_validator(mode="after")
    def _check_v24(self) -> "Settings":
        """V2.4 配置校验：只告警不抛异常（老 .env 仍可启动，§2.11）。"""
        if not 0 <= self.trace_sample_rate <= 1:
            logger.warning(
                "trace_sample_rate=%s 越界（应在 0~1），将按 1.0 处理",
                self.trace_sample_rate,
            )
        if self.llm_max_concurrency < 1:
            logger.warning(
                "llm_max_concurrency=%s < 1，将按 1 处理", self.llm_max_concurrency
            )
        if self.llm_timeout_s <= 0:
            logger.warning("llm_timeout_s=%s <= 0，将按 30s 处理", self.llm_timeout_s)
        if self.reminder_pending_ttl_hours <= 0:
            logger.warning(
                "reminder_pending_ttl_hours=%s <= 0，将按 24h 处理",
                self.reminder_pending_ttl_hours,
            )
        if self.db_pool_min_size < 1:
            logger.warning("db_pool_min_size=%s < 1，将按 1 处理", self.db_pool_min_size)
        if self.db_pool_max_size < max(1, self.db_pool_min_size):
            logger.warning(
                "db_pool_max_size=%s < db_pool_min_size=%s，将按 min_size 处理",
                self.db_pool_max_size,
                self.db_pool_min_size,
            )
        return self

    @model_validator(mode="after")
    def _check_summary_window(self) -> "Settings":
        """V2.3（D9）：摘要触发轮与近轮窗口对齐校验；只告警，不抛异常。"""
        if self.summary_trigger_rounds > self.prompt_recent_rounds:
            logger.warning(
                "summary_trigger_rounds=%s > prompt_recent_rounds=%s："
                "第 %s–%s 轮可能掉出 prompt 窗口却尚未摘要，形成上下文空窗",
                self.summary_trigger_rounds,
                self.prompt_recent_rounds,
                self.prompt_recent_rounds + 1,
                self.summary_trigger_rounds,
            )
        if self.summary_trigger_rounds < self.summary_chunk_rounds:
            logger.warning(
                "summary_trigger_rounds=%s < summary_chunk_rounds=%s："
                "每轮摘要覆盖的轮次多于触发间隔，可能重复压缩",
                self.summary_trigger_rounds,
                self.summary_chunk_rounds,
            )
        return self

    @staticmethod
    def _to_psycopg(url: str) -> str:
        """SQLAlchemy 走 psycopg3 驱动：postgresql:// -> postgresql+psycopg://。"""
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://") :]
        return url

    @property
    def sqlalchemy_url(self) -> str:
        return self._to_psycopg(self.database_url)

    @property
    def memory_api_agent_set(self) -> frozenset[str]:
        """M2：`MEMORY_API_AGENTS` 解析为白名单集合。

        逗号分隔、strip、去空；**纯数字一律剔除**（会与角色 `agent:1` namespace 撞车）。
        空白名单 → 空集合（所有 agent 403，启动时 warning）。
        """
        out: set[str] = set()
        for part in str(self.memory_api_agents or "").split(","):
            name = part.strip()
            if name and not name.isdigit():
                out.add(name)
        return frozenset(out)

    @property
    def test_sqlalchemy_url(self) -> str:
        """测试库 SQLAlchemy 连接串（真库集成测试用）。"""
        return self._to_psycopg(self.test_database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
