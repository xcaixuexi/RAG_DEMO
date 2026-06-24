"""
view/web_view.py — FastAPI Web 层

接口：
    POST /api/chat              文字对话（首次查询，返回第 1 页）
    POST /api/file/upload       简历文件上传解析
    GET  /api/jobs/page         职位/候选人/统计列表翻页（从 session 缓存切片，不重查 DB）
    GET  /api/stats             路由命中率统计

v2 变更：
    - ChatRequest 新增可选字段 tenant_id（admin 登录时由前端传入，后端鉴权时自动覆盖）
    - /api/chat 透传 user_role 给 ChatController，admin 鉴权逻辑在 Controller 层执行
    - 收到 auth_failed 响应时，HTTP 状态码仍为 200，由前端根据 intent 字段判断并跳转

v3 变更（统一响应规范）：
    - /api/jobs/page 翻页接口支持 result_type="stats_rows"（platform_stats 列表翻页）
    - 翻页响应统一使用 items 字段，去掉 data_key 硬编码
    - intent 根据 result_type 自动推断
"""

import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from controller.chat_controller import ChatController
from controller import session_manager

logger = logging.getLogger(__name__)

# ── 临时文件目录，启动时自动创建 ──────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _setup_logging():
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    if root.handlers:   # 防止 reload 时重复添加
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                            datefmt="%H:%M:%S")
    for handler in [logging.StreamHandler(),
                    logging.FileHandler("logs/app.log", encoding="utf-8")]:
        handler.setFormatter(fmt)
        root.addHandler(handler)

_setup_logging()

app = FastAPI(title="招聘AI助手", version="1.0.0")

# ── CORS（开发阶段放开，生产按需收紧）────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 请求模型
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id:    int = 161102110337
    user_role:  str                  # recruiter | jobseeker | admin
    session_id: str = "3a7b9f2c-8d4e-4a1b-9c6d-7e8f2a1b3c4d"
    message:    str
    page_size:  int = 20
    tenant_id:  Optional[int] = None


# ─────────────────────────────────────────────
# result_type → intent 映射
# ─────────────────────────────────────────────

_RESULT_TYPE_TO_INTENT = {
    "jobs":       "job_search",
    "candidates": "candidate_query",
    "stats_rows": "platform_stats",
}

_VALID_RESULT_TYPES = set(_RESULT_TYPE_TO_INTENT.keys())


# ─────────────────────────────────────────────
# 接口实现
# ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    """
    文字对话接口。
    职位/候选人类查询：返回第 1 页数据 + pagination 元信息。
    其余意图：返回文本 message，items / pagination 均为 null。\n
    recruiter 招聘者\n
    jobseeker 求职者\n
    admin     user_id=88926257 平台管理员（需在后端完成 tenant_id 鉴权）\n
    session_id 对话id\n
    """
    controller = ChatController(
        user_role  = request.user_role,
        user_id    = request.user_id,
        session_id = request.session_id,
    )
    return controller.process_message(request.message)


@app.get("/api/jobs/page")
async def jobs_page(
    session_id:  str = Query(...,          description="会话 ID"),
    result_type: str = Query("jobs",       description="缓存类型：jobs | candidates | stats_rows"),
    page:        int = Query(1,    ge=1,   description="页码，从 1 开始"),
    page_size:   int = Query(20,   ge=1,  le=100, description="每页条数，最大 100"),
) -> dict:
    """
    翻页接口：从 session 内存缓存切片返回，不重新查询数据库。

    前端使用流程：
        1. POST /api/chat  →  获得第 1 页数据和 pagination 元信息
        2. 用户点击"下一页"  →  GET /api/jobs/page?session_id=xxx&page=2
        3. 缓存过期（session 超时）时返回 404，前端引导用户重新查询

    result_type 说明：
        "jobs"       对应 job_search / job_manage 的职位列表
        "candidates" 对应 candidate_search 的候选人列表
        "stats_rows" 对应 platform_stats LLM 路径的列表查询结果

    注意：count-only 查询（pagination 为 null）不写缓存，
    前端无需调用此接口，直接展示 message 中的总数即可。
    """
    if result_type not in _VALID_RESULT_TYPES:
        raise HTTPException(
            status_code = 400,
            detail      = f"result_type 只能是 {' / '.join(_VALID_RESULT_TYPES)}",
        )

    page_data = session_manager.get_page(
        session_id  = session_id,
        result_type = result_type,
        page        = page,
        page_size   = page_size,
    )

    if page_data is None:
        raise HTTPException(
            status_code = 404,
            detail      = "查询缓存已过期，请重新发起查询",
        )

    intent = _RESULT_TYPE_TO_INTENT[result_type]

    return {
        "intent": intent,
        "data": {
            "message":    f"第 {page_data['page']} 页 / 共 {page_data['total_pages']} 页",
            "total":      page_data["total_db"],
            "list_type":  result_type,
            "items":      page_data["items"],
            "pagination": {
                "page":        page_data["page"],
                "page_size":   page_data["page_size"],
                "total_pages": page_data["total_pages"],
                "fetched":     page_data["fetched"],
                "total_db":    page_data["total_db"],
            },
        },
        "status": "success",
    }


@app.post("/api/file/upload")
async def upload_file(
    file:       UploadFile = File(...),
    user_id:    int        = Form(...),
    user_role:  str        = Form("recruiter"),
    session_id: str        = Form(...),
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
            detail=f"不支持的文件类型 '{suffix}'，请上传 .pdf 或 .docx 文件"
        )

    # 保存临时文件
    safe_name = f"{user_id}_{int(time.time())}_{file.filename}"
    temp_path = UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        temp_path.write_bytes(content)
        logger.info(f"临时文件已保存: {temp_path}（{len(content)} bytes）")
        controller = ChatController(
            user_role  = user_role,
            user_id    = user_id,
            session_id = session_id,
        )
        response = controller.process_file(str(temp_path))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        raise HTTPException(status_code=500, detail="文件处理失败，请稍后重试")
    finally:
        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"临时文件已删除: {temp_path}")

    return response


@app.get("/api/stats")
async def stats() -> dict:
    """路由命中率 + session 状态统计"""
    controller = ChatController(session_id="__stats__")
    return {
        "routing": controller.get_routing_stats(),
        "session": session_manager.stats(),
    }
