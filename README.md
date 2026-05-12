### 环境变量

- `LLM_BASE_URL`: LLM服务地址. 如 DeepSeek openai 适配访问 baseUrl 为 `https://api.deepseek.com`
- `LLM_API_KEY`: LLM apiKey
- `LLM_MODEL`: LLM 模型
- `LLM_PROVIDER`: LLM 服务商. 默认为 `openai`, 目前只支持 `openai`
- `WORK_DIR`: 工作目录. 默认为 `src/main.py` 所在的目录
- `ENABLE_THINKING`: 是否开启 two step 思考. 默认为 `false`