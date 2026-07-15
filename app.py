"""
农业植保助手 — FastAPI 后端服务
启动方式: python app.py
浏览器访问: http://localhost:8000
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
# 知识库（可选）
try:
    from knowledge_base import search_knowledge, build_knowledge_base
except ImportError:
    search_knowledge = lambda *a, **kw: []
    build_knowledge_base = lambda: None

# LangChain（可选，无则用 httpx 直调）
try:
    from langchain_handler import chat_stream_with_rag
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    chat_stream_with_rag = None

# ============================================================
# 配置
# ============================================================
load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-vl-plus"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "conversations"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "你是农业植保AI助手，为农民朋友提供专业的农业技术服务。\n\n"
    "专业范围：作物病虫害识别与防治、农药配比与安全使用、种植技术指导、"
    "土壤改良与施肥、果蔬种植管理。\n\n"
    "回答要求：\n"
    "- 语言通俗，像村里农技员说话一样\n"
    "- 如果用户发了照片，优先分析照片里的作物症状\n"
    "- 如果「知识库参考」中有相关内容，优先基于知识库内容给出诊断和方案\n"
    "- 推荐农药必须说清：药名、兑水比例、怎么打、几天打一次、最后一次打离采收几天\n"
    "- 不确定的要说「建议拿样品去当地农技站看看」\n"
    "- 用短句、分点说，方便阅读\n"
    "- 每条回复不要太长，说重点就行"
)

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="农业植保助手", version="2.0")


# ============================================================
# 数据模型
# ============================================================
class ChatRequest(BaseModel):
    user_id: str
    conversation_id: str
    message: str = ""
    image: Optional[str] = None


class ConversationCreate(BaseModel):
    title: str = "新对话"


class SpeechRequest(BaseModel):
    audio: str  # base64 音频数据


# ============================================================
# 对话存储（按用户隔离，JSON 文件）
# ============================================================
def _user_dir(user_id: str) -> Path:
    d = DATA_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_index_path(user_id: str) -> Path:
    return _user_dir(user_id) / "_index.json"


def _load_index(user_id: str) -> list:
    p = _user_index_path(user_id)
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return []
    return []


def _save_index(user_id: str, data: list) -> None:
    _user_index_path(user_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _conv_path(user_id: str, conv_id: str) -> Path:
    return _user_dir(user_id) / f"{conv_id}.json"


def _load_conv(user_id: str, conv_id: str) -> dict:
    path = _conv_path(user_id, conv_id)
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: pass
    return {"id": conv_id, "title": "新对话", "created_at": datetime.now().isoformat(), "messages": []}


def _save_conv(user_id: str, conv: dict) -> None:
    _conv_path(user_id, conv["id"]).write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")


def _add_message(user_id: str, conv_id: str, role: str, content: str, thumb: str = "") -> dict:
    conv = _load_conv(user_id, conv_id)
    msg = {"role": role, "content": content}
    if thumb: msg["thumb"] = thumb
    conv["messages"].append(msg)
    if conv["title"] == "新对话" and role == "user":
        text = content if isinstance(content, str) else (content[0].get("text", "") if isinstance(content, list) else "")
        conv["title"] = text[:20] if text else "新对话"
    _save_conv(user_id, conv)
    _update_index_entry(user_id, conv)
    return conv


def _update_index_entry(user_id: str, conv: dict) -> None:
    idx = _load_index(user_id)
    for e in idx:
        if e["id"] == conv["id"]:
            e["title"] = conv["title"]
            e["message_count"] = len(conv["messages"])
            _save_index(user_id, idx)
            return
    idx.insert(0, {"id": conv["id"], "title": conv["title"], "created_at": conv["created_at"], "message_count": len(conv["messages"])})
    _save_index(user_id, idx)


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health")
async def health():
    return {"status": "ok", "api_key_configured": bool(API_KEY)}


# --- 用户注册/检查 ---
@app.get("/api/user-exists")
async def user_exists(user_id: str):
    """检查用户是否已注册"""
    return {"exists": _user_index_path(user_id).exists()}


@app.post("/api/register")
async def register(user_id: str):
    """注册新用户"""
    if _user_index_path(user_id).exists():
        raise HTTPException(status_code=409, detail="用户已存在")
    _save_index(user_id, [])
    return {"ok": True, "user_id": user_id}


# --- 对话列表 ---
@app.get("/api/conversations")
async def list_conversations(user_id: str):
    return _load_index(user_id)


@app.post("/api/conversations")
async def create_conversation(body: ConversationCreate, user_id: str):
    conv_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
    conv = {"id": conv_id, "title": body.title, "created_at": datetime.now().isoformat(), "messages": []}
    _save_conv(user_id, conv)
    _update_index_entry(user_id, conv)
    return conv


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, user_id: str):
    conv = _load_conv(user_id, conv_id)
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user_id: str):
    path = _conv_path(user_id, conv_id)
    if path.exists(): path.unlink()
    idx = _load_index(user_id)
    idx = [e for e in idx if e["id"] != conv_id]
    _save_index(user_id, idx)
    return {"ok": True}


# --- 聊天（流式） ---
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """流式聊天：代理 DashScope API，返回 SSE 流"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未配置，请设置 .env 文件")

    # 构建消息列表
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # RAG：检索病虫害知识库
    knowledge_text = request.message or "病虫害"
    kb_results = search_knowledge(knowledge_text, n_results=3)
    if kb_results:
        kb_context = "\n\n---\n\n".join([
            f"【知识库匹配 {i+1}】作物：{r['metadata'].get('crop','')} | 病害：{r['metadata'].get('disease','')} | 相关度：{r['score']}\n{r['document']}"
            for i, r in enumerate(kb_results)
        ])
        messages.append({"role": "system", "content": f"【知识库参考】以下是与你当前查询最相关的病虫害资料，请参考这些信息回答：\n\n{kb_context}"})

    # 加载历史（最近 20 条）
    conv = _load_conv(request.user_id, request.conversation_id)
    history = conv.get("messages", [])[-20:]
    for msg in history:
        # 缩略图不发送给 API（只发文字）
        if isinstance(msg.get("content"), str):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # 构建用户消息
    if request.image:
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": request.message or "请帮我看看这张作物照片，判断是什么病虫害，该怎么治？"},
                {"type": "image_url", "image_url": {"url": request.image}},
            ],
        }
        _add_message(request.user_id, request.conversation_id, "user",
                     request.message or "我发了一张作物照片",
                     thumb=request.image)
    else:
        user_msg = {"role": "user", "content": request.message}
        _add_message(request.user_id, request.conversation_id, "user", request.message)

    messages.append(user_msg)

    # 流式生成（LangChain 优先，httpx 兜底）
    history_messages = conv.get("messages", [])[-20:]

    if HAS_LANGCHAIN:
        async def stream_gen():
            full = ""
            try:
                async for chunk in chat_stream_with_rag(
                    question=request.message or "请帮我看看这张作物照片",
                    history=history_messages
                ):
                    full += chunk
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
                if full:
                    _add_message(request.user_id, request.conversation_id, "assistant", full)
    else:
        async def stream_gen():
            full = ""
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST", f"{ENDPOINT.rstrip('/')}/chat/completions",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
                        json={"model": MODEL, "messages": messages, "stream": True, "max_tokens": 1500, "temperature": 0.7},
                    ) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            yield f"data: {json.dumps({'error': f'API {resp.status_code}'})}\n\ndata: [DONE]\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if line and line.startswith("data:"):
                                yield f"{line}\n\n"
                                data_str = line[5:].strip()
                                if data_str != "[DONE]":
                                    try:
                                        d = json.loads(data_str)
                                        full += d["choices"][0]["delta"].get("content", "")
                                    except Exception: pass
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\ndata: [DONE]\n\n"
            finally:
                yield "data: [DONE]\n\n"
                if full:
                    _add_message(request.user_id, request.conversation_id, "assistant", full)

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# --- 语音识别（DashScope paraformer，全浏览器兼容） ---
@app.post("/api/speech-recognize")
async def speech_recognize(body: SpeechRequest):
    """调用 DashScope 语音识别，返回文字"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "paraformer-v1",
                    "input": {"audio": body.audio},
                    "parameters": {
                        "format": "wav",
                        "sample_rate": 16000,
                        "language_hints": ["zh", "en"],
                    },
                },
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"语音识别失败: {resp.text[:200]}")
            data = resp.json()
            text = data.get("output", {}).get("text", "")
            return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别异常: {str(e)}")


# ============================================================
# 静态文件（前端界面）
# ============================================================
@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")


# ============================================================
# 启动时迁移旧数据 + 初始化知识库
# ============================================================
def _migrate_old_data():
    legacy = DATA_DIR / "_legacy"
    old_files = list(DATA_DIR.glob("*.json"))
    if not old_files: return
    legacy.mkdir(parents=True, exist_ok=True)
    for f in old_files:
        if f.name == "_index.json": continue
        try: f.rename(legacy / f.name)
        except Exception: pass
    old_idx = DATA_DIR / "_index.json"
    if old_idx.exists():
        try: old_idx.rename(legacy / "_index.json")
        except Exception: pass
    print(f"  已迁移 {len(old_files)} 个旧对话到 legacy 用户")

_migrate_old_data()

# 初始化病虫害知识库
print("  初始化病虫害知识库...")
try:
    build_knowledge_base()
except Exception as e:
    print(f"  知识库初始化失败（将跳过RAG增强）: {e}")


# --- 知识库搜索 API ---
@app.get("/api/knowledge-search")
async def knowledge_search(q: str, n: int = 3):
    """搜索病虫害知识库"""
    return search_knowledge(q, n_results=n)

# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    print(f"""
============================================
  农业植保助手 v2.0
  FastAPI 前后端分离架构
============================================
  后端地址: http://{HOST}:{PORT}
  API 文档: http://localhost:{PORT}/docs
  API Key:  {'[已配置]' if API_KEY else '[未配置! 请编辑 .env 文件]'}
============================================
    """.strip())

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
