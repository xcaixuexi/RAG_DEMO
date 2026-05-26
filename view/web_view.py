"""
view/web_view.py — FastAPI Web 层

接口：
    POST /api/chat          文字对话
    POST /api/file/upload   简历文件上传解析
    GET  /api/stats         路由命中率统计
"""

import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from controller.chat_controller import ChatController

logger = logging.getLogger(__name__)

# ── 临时文件目录，启动时自动创建 ──────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="招聘AI助手", version="1.0.0")

# ── CORS（开发阶段放开，生产按需收紧）────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 请求 / 响应模型
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id:   int
    user_role: str   # recruiter | jobseeker
    message:   str


# ─────────────────────────────────────────────
# 接口实现
# ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    """
    文字对话接口。

    每次请求独立初始化 ChatController（无状态，user_role/user_id 从请求体取）。\n
    recruiter 招聘者\n
    jobseeker 求职者
    """
    controller = ChatController(
        user_role=request.user_role,
        user_id=request.user_id,
    )
    response = controller.process_message(request.message)
    return response


@app.post("/api/file/upload")
async def upload_file(
    file:      UploadFile = File(...),
    user_id:   int        = Form(...),
    user_role: str        = Form("recruiter"),
) -> dict:
    """
    简历文件上传解析接口。

    流程：
        接收 UploadFile → 保存临时文件 → 调用 process_file() → 删除临时文件 → 返回响应
    文件名格式：{user_id}_{timestamp}_{原文件名}，避免并发冲突。
    """
    # 校验文件类型
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{suffix}'，请上传 .pdf 或 .docx 文件",
        )

    # 保存临时文件
    safe_name   = f"{user_id}_{int(time.time())}_{file.filename}"
    temp_path   = UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        temp_path.write_bytes(content)
        logger.info(f"临时文件已保存: {temp_path}（{len(content)} bytes）")

        # 调用 Controller 解析
        controller = ChatController(user_role=user_role, user_id=user_id)
        response   = controller.process_file(str(temp_path))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        raise HTTPException(status_code=500, detail="文件处理失败，请稍后重试")
    finally:
        # 无论成功与否，临时文件都要清理
        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"临时文件已删除: {temp_path}")

    return response


@app.get("/api/stats")
async def stats() -> dict:
    """
    路由命中率统计接口（用于监控规则路由效果）。

    注意：每次请求会新建一个 Controller，stats 从 0 开始——
    生产环境若需全局统计，应将 Supervisor 改为单例。
    当前阶段用于开发调试。
    """
    controller = ChatController()
    return controller.get_routing_stats()
