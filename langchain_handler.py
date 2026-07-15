"""
LangChain 集成模块 — 按课程 Day 9-12 教学实现
包含：ChatModel、PromptTemplate、LCEL Chain、ReAct Agent
"""
import os
import json
from typing import List, Dict, Any, Generator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from knowledge_base import search_knowledge

# ============================================================
# 模型配置（DashScope 兼容 OpenAI API）
# ============================================================
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", API_KEY)

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-plus"

_llm = None
_llm_stream = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7, max_tokens=1500)
    return _llm


def _get_llm_stream():
    global _llm_stream
    if _llm_stream is None:
        _llm_stream = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7, max_tokens=1500, streaming=True)
    return _llm_stream

# ============================================================
# Prompt 模板（Day 4 & Day 9）
# ============================================================
SYSTEM_TEMPLATE = (
    "你是农业植保AI助手，为农民朋友提供专业的农业技术服务。\n\n"
    "专业范围：作物病虫害识别与防治、农药配比与安全使用、种植技术指导、"
    "土壤改良与施肥、果蔬种植管理。\n\n"
    "回答要求：\n"
    "- 语言通俗，像村里农技员说话一样\n"
    "- 如果用户发了照片，优先分析照片里的作物症状\n"
    "- 如果「知识库参考」中有相关内容，必须基于知识库内容给出诊断和方案\n"
    "- 推荐农药必须说清：药名、兑水比例、怎么打、几天打一次、最后一次打离采收几天\n"
    "- 不确定的要说「建议拿样品去当地农技站看看」\n"
    "- 用短句、分点说，方便阅读\n"
    "- 每条回复不要太长，说重点就行"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# ============================================================
# RAG Chain（Day 8-9：检索增强生成）
# ============================================================
def retrieve_context(question: str) -> str:
    """从 ChromaDB 知识库检索相关病虫害信息"""
    results = search_knowledge(question, n_results=3)
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results):
        parts.append(
            f"【知识库匹配 {i+1}】作物：{r['metadata'].get('crop','')} | "
            f"病害：{r['metadata'].get('disease','')} | 相关度：{r['score']}\n"
            f"{r['document']}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_chain():
    """构建 RAG 链：检索 → 注入上下文 → LLM 生成"""
    def prepare_input(data: dict) -> dict:
        question = data.get("question", "")
        history = data.get("history", [])
        context = retrieve_context(question)
        if context:
            # 在 question 前注入知识库内容
            enhanced = f"【知识库参考】\n{context}\n\n【用户问题】\n{question}"
        else:
            enhanced = question
        return {"question": enhanced, "history": history}

    chain = (
        RunnableLambda(prepare_input)
        | prompt_template
        | _get_llm()
        | StrOutputParser()
    )
    return chain


# ============================================================
# Agent 工具（Day 10-11：工具调用 + ReAct Agent）
# ============================================================
@tool
def search_crop_disease(query: str) -> str:
    """搜索农作物病虫害知识库。当用户描述作物症状、询问病虫害防治方法、或需要农药配比信息时使用。
    参数 query: 搜索关键词，如'玉米黄斑'、'水稻瘟病'等"""
    results = search_knowledge(query, n_results=3)
    if not results:
        return "未找到相关病虫害信息，建议拿样品去当地农技站看看。"
    out = []
    for i, r in enumerate(results):
        out.append(
            f"### 结果{i+1}：{r['metadata'].get('crop','')} - {r['metadata'].get('disease','')}（相关度：{r['score']}）\n"
            f"{r['document']}"
        )
    return "\n\n".join(out)


@tool
def calculate_pesticide_ratio(water_amount_liters: float, dilution_ratio: int) -> str:
    """计算农药兑水配比。当用户询问'多少药兑多少水'时使用。
    参数 water_amount_liters: 需要的水量（升）
    参数 dilution_ratio: 稀释倍数，如800倍液则填800"""
    pesticide_ml = (water_amount_liters * 1000) / dilution_ratio
    return (
        f"配比计算结果：\n"
        f"- 水量：{water_amount_liters}升\n"
        f"- 稀释倍数：{dilution_ratio}倍\n"
        f"- 需要农药：{pesticide_ml:.1f}毫升（约{pesticide_ml/1000*1000:.0f}克）\n"
        f"- 用法：将{pesticide_ml:.1f}毫升农药加入{water_amount_liters}升水中，搅拌均匀后喷雾\n"
        f"- 提示：先用少量水稀释农药，再加满水搅拌均匀"
    )


# Agent 工具集
AGENT_TOOLS = [search_crop_disease, calculate_pesticide_ratio]

# 创建 ReAct Agent（Day 11）
def build_react_agent():
    """构建 ReAct Agent：Thought → Action → Observation 循环"""
    return create_react_agent(
        model=_get_llm(),
        tools=AGENT_TOOLS,
        prompt=(
            "你是农业植保AI助手。\n"
            "当用户询问病虫害相关问题时，必须先调用 search_crop_disease 工具搜索知识库。\n"
            "当用户需要计算农药配比时，调用 calculate_pesticide_ratio 工具。\n"
            "回答要求：\n"
            "- 语言通俗，像村里农技员说话一样\n"
            "- 推荐农药必须说清：药名、兑水比例、怎么打、几天打一次\n"
            "- 不确定的要说「建议拿样品去当地农技站看看」\n"
            "- 用短句、分点说"
        ),
    )


# ============================================================
# 统一聊天接口（支持流式 + Agent 模式）
# ============================================================
def chat_with_rag(question: str, history: List[Dict] = None) -> str:
    """RAG 模式：检索 + LLM 生成（非流式）"""
    if history is None:
        history = []
    chain = build_rag_chain()
    return chain.invoke({"question": question, "history": history})


async def chat_stream_with_rag(question: str, history: List[Dict] = None):
    """RAG 模式：流式输出（异步生成器）"""
    if history is None:
        history = []

    context = retrieve_context(question)
    if context:
        enhanced = f"【知识库参考】\n{context}\n\n【用户问题】\n{question}"
    else:
        enhanced = question

    msgs = [SystemMessage(content=SYSTEM_TEMPLATE)]
    for h in history or []:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"] if isinstance(h["content"], str) else h["content"].get("text", "")))
        elif h["role"] == "assistant":
            msgs.append(AIMessage(content=h["content"]))
    msgs.append(HumanMessage(content=enhanced))

    async for chunk in _get_llm_stream().astream(msgs):
        delta = chunk.content if hasattr(chunk, 'content') else str(chunk)
        if delta:
            yield delta


def chat_with_agent(question: str) -> str:
    """Agent 模式：ReAct 推理 + 工具调用"""
    agent = build_react_agent()
    result = agent.invoke({"messages": [("user", question)]})
    # 提取最后一条 AI 消息
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'content') and msg.type == 'ai':
            return msg.content
    return "抱歉，处理出错，请重试。"