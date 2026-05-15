# tiny-claw

一个基于钉钉消息入口的轻量 Agent Bot。

## Quickstart

### 1) 前置条件

- Python `>=3.13`
- `uv`（用于依赖管理与运行）

### 2) 安装依赖

```bash
uv sync
```

### 3) 配置环境变量

在项目根目录创建 `.env`（可从 `src/.env.example` 复制）：

```bash
cp src/.env.example .env
```

按需填写变量，至少保证必填项不为空。

### 4) 启动

```bash
uv run python src/main.py
```

## 环境变量

### 必填

- `DINGTALK_CLIENT_ID`：钉钉客户端 ID
- `DINGTALK_CLIENT_SECRET`：钉钉客户端密钥
- `LLM_API_KEY`：模型 API Key

### 常用可选

- `LLM_PROVIDER`：LLM 提供商，默认 `openai`
- `LLM_BASE_URL`：LLM Base URL，默认 `https://api.openai.com/v1`
- `LLM_MODEL`：模型名，默认 `gpt-5.3-codex`

### 运行参数

- `WORK_DIR`：工作目录，默认当前进程工作目录
- `ENABLE_THINKING`：是否启用思考模式，默认 `false`
- `BASH_TIMEOUT`：Bash 工具超时时间（秒），默认 `30`

## 常见问题

- 启动时报配置错误：检查必填变量是否为空。
- 模型请求失败：检查 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 是否匹配。
- 钉钉侧无响应：检查 `DINGTALK_CLIENT_ID` 和 `DINGTALK_CLIENT_SECRET` 是否正确。
