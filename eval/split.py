"""确定性 dev/test 切分（契约 4.4）：同一 id 永远同一侧，跨机器一致。"""

import hashlib

SEED = 20261015


def split_of(case_id: str) -> str:
    """sha256(f"{SEED}:{case_id}") 取前 8 字节 → int → % 100 < 70 为 'dev'，否则 'test'。"""
    digest = hashlib.sha256(f"{SEED}:{case_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "dev" if bucket < 70 else "test"
