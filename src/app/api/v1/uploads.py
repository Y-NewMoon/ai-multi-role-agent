"""文件上传（M5）：角色头像。

边界校验（系统边界，必须校验）：按文件头 magic bytes 判定真实图片类型，
忽略客户端声明的 content-type；仅允许 png/jpg/webp/gif，单文件上限见 settings.avatar_max_bytes。
返回值为可直接写入 `agents.avatar` 的静态 URL（形如 /static/avatars/<uuid>.png）。
"""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

router = APIRouter(prefix="/uploads", tags=["uploads"])

_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
)


def _sniff_ext(data: bytes) -> str | None:
    """按文件头识别图片类型，识别不了返回 None。"""
    for sig, ext in _SIGNATURES:
        if data.startswith(sig):
            return ext
    # WEBP：RIFF....WEBP
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


@router.post("/avatar")
async def upload_avatar(request: Request, file: UploadFile = File(...)) -> dict[str, str]:
    settings = request.app.state.settings

    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="文件为空")
    if len(data) > settings.avatar_max_bytes:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"图片过大（上限 {settings.avatar_max_bytes // (1024 * 1024)}MB）",
        )

    ext = _sniff_ext(data)
    if ext is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="仅支持 png/jpg/webp/gif 图片")

    dir_path = Path(settings.avatar_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    name = f"{uuid4().hex}{ext}"
    (dir_path / name).write_bytes(data)
    return {"url": f"{settings.avatar_url_prefix}/{name}"}
