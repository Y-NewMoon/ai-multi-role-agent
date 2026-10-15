"""应用配置：从 .env 读取（pydantic-settings）。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    summary_trigger_rounds: int = 20
    summary_chunk_rounds: int = 10

    # ===== 一次回复的消息条数上限（M5：角色可连发 1~N 条）=====
    reply_max_messages: int = 3

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
    def test_sqlalchemy_url(self) -> str:
        """测试库 SQLAlchemy 连接串（真库集成测试用）。"""
        return self._to_psycopg(self.test_database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
