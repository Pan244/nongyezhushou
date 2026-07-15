# 农业植保助手

FastAPI + LangChain + ChromaDB + RAG + ReAct Agent 前后端分离项目

## 环境要求
- Python 3.8+
- Windows / Mac / Linux

## 换电脑使用步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key
编辑 `.env` 文件，填入 DashScope API Key：
```
DASHSCOPE_API_KEY=你的key
```

### 3. 启动
```bash
python app.py
```

### 4. 打开浏览器
访问 `http://localhost:8000`

> 如果端口被占用，编辑 `.env` 文件改 `PORT=8080`

## 同 WiFi 分享
启动后，其他人访问 `http://你的电脑IP:8000`
查看本机 IP：`ipconfig`（Windows）或 `ifconfig`（Mac）

## 项目结构
```
├── app.py              FastAPI 后端
├── langchain_handler.py LangChain RAG + Agent
├── knowledge_base.py   ChromaDB 病虫害知识库
├── static/index.html   前端界面
├── .env                API Key 配置
├── requirements.txt    依赖清单
└── 启动助手.bat         Windows 一键启动
```
