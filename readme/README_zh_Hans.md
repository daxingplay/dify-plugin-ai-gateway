# AI 网关 Dify 插件

这是一个 Dify 插件,用于集成[阿里云 AI 网关](https://help.aliyun.com/zh/api-gateway/ai-gateway/product-overview/what-is-an-ai-gateway),通过网关的协议转换、安全防护、流量治理和可观测性功能访问 AI 模型。

## 概述

本插件允许 Dify 连接阿里云 AI 网关,AI 网关作为 AI 应用与模型服务之间的核心连接组件,提供以下能力:

- **协议转换**: 统一的接口访问各类 AI 模型
- **安全防护**: 内容审核、提示词注入检测和敏感数据过滤
- **流量治理**: 限流、负载均衡和智能路由
- **统一可观测**: 监控、日志和 Token 用量跟踪
- **成本优化**: 请求缓存和基于 Token 的计费

## 什么是阿里云 AI 网关?

[阿里云 AI 网关](https://help.aliyun.com/zh/api-gateway/ai-gateway/product-overview/what-is-an-ai-gateway)是一个用于 AI 应用集成的网关服务,提供以下功能:

### 核心能力

1. **多模型支持**: 统一接口访问不同供应商的模型(OpenAI、Anthropic、阿里通义千问等)
2. **多模态支持**: 处理文本、图像、音频和视频,支持多种传输协议
3. **安全合规**: 内置 AI 安全功能,包括:
   - 内容审核和合规检查
   - 提示词注入攻击防护
   - 敏感数据检测
   - 恶意文件检测
   - 数字水印标识
4. **流量治理**: 
   - 基于消费者的限流
   - Token 消耗限制
   - 并发请求控制
   - 智能负载均衡
5. **可观测性和分析**:
   - 实时请求/响应监控
   - Token 消耗统计
   - 缓存命中率跟踪
   - 模型性能指标

## 功能特性

### 支持的模型类型

- ✅ **大语言模型(LLM)**: 对话和补全模型
- ✅ **重排序(Rerank)**: 文档重排序模型
- ✅ **语音转文本**: 音频转录模型
- ✅ **文本嵌入**: 向量嵌入模型
- ✅ **文本转语音**: 语音合成模型

### 认证方式

本插件支持三种 AI 网关认证方式:

1. **API Key**: 简单的 Bearer Token 认证
2. **JWT(JSON Web Token)**: 基于标准的认证,支持 JWKS
3. **HMAC**: 使用访问密钥和密钥对请求签名

### 其他功能

- **结构化输出**: LLM 响应的 JSON Schema 验证
- **思考模式**: 支持具有推理能力的模型(o1、DeepSeek R1 等)
- **自定义模型配置**: 每个模型的显示名称和参数
- **流式支持**: 所有兼容模型的实时流式响应

## 安装

### 前置条件

- Dify 实例(自托管或云版本)
- 已启用 AI 网关的阿里云账号
- Python 3.11+(用于开发)

### 安装插件

1. **从 Dify 市场安装**(推荐):
   - 进入您的 Dify 实例
   - 前往 **插件** → **市场**
   - 搜索 "AI Gateway"
   - 点击 **安装**

2. **手动安装**(开发):
   ```bash
   # 克隆仓库
   git clone <repository-url>
   cd dify-plugin-ai-gateway
   
   # 安装依赖
   pip install -r requirements.txt
   
   # 打包插件
   dify-plugin plugin package .
   
   # 上传 .difypkg 文件到您的 Dify 实例
   ```

## 配置

### 步骤 1: 创建 AI 网关消费者

使用本插件前,您需要在阿里云 AI 网关创建一个消费者。请参考[创建消费者官方指南](https://help.aliyun.com/zh/api-gateway/ai-gateway/user-guide/creating-a-consumer)。

消费者代表一个可以访问 AI 网关 API 的身份,用于认证、授权和计费。

### 步骤 2: 配置认证

在 Dify 中添加模型时,您需要提供以下凭证:

#### 通用设置

- **端点 URL**: 您的 AI 网关端点(例如: `https://your-gateway-cn-hangzhou.alicloudapi.com/v1`)
- **模型名称**: AI 网关中的模型标识符(例如: `qwen-turbo`、`qwen-max`)
- **显示名称**: (可选)在 Dify UI 中显示的自定义名称

#### 认证方式 1: API Key

```yaml
认证方式: api_key
API Key: 来自 AI 网关的 API 密钥
```

**使用场景**: 简单场景,您控制 API 密钥的分发。

#### 认证方式 2: JWT(JSON Web Token)

```yaml
认证方式: jwt
JWT JWKS: {"k":"your-base64-key","kty":"oct","alg":"HS256"}
JWT 算法: HS256(或 RS256、ES256 等)
JWT 消费者 ID: 您的消费者 ID
JWT 过期时间: 7200(秒)
JWT 签发者: (可选)您的签发者
JWT 头名称: Authorization
JWT 头前缀: Bearer
```

**使用场景**: 企业场景,需要基于标准的令牌认证和自动过期。

**JWKS 格式**: JWKS(JSON Web Key Set)应包含您的签名密钥。对于 HMAC 算法(HS256/384/512),使用:
```json
{
  "k": "base64编码的密钥",
  "kty": "oct",
  "alg": "HS256"
}
```

#### 认证方式 3: HMAC 签名

```yaml
认证方式: hmac
HMAC Access Key: 您的访问密钥
HMAC Secret Key: 您的密钥
```

**使用场景**: 需要请求完整性验证的场景。每个请求都使用时间戳和内容哈希进行签名。

**HMAC 工作原理**:
1. 插件从请求构造规范字符串(方法、路径、头、正文)
2. 使用您的密钥通过 SHA256 对字符串签名
3. 在请求头中包含签名、访问密钥和时间戳
4. AI 网关验证签名以认证请求

### 步骤 3: 配置模型参数

根据模型类型,您可以配置其他参数:

**对于 LLM 模型**:
- Temperature、Top P、Top K
- 最大 Token 数
- 响应格式(text、json_object、json_schema)
- 思考模式(用于推理模型)
- 停止序列

**对于嵌入模型**:
- 文档前缀
- 查询前缀
- 嵌入维度

**对于语音模型**:
- 语言
- 语音选择
- 音频格式

## 使用示例

### 在 Dify 聊天机器人中使用

1. 在 Dify 中创建新的聊天机器人应用
2. 在模型设置中选择您的 AI 网关模型
3. 配置系统提示词和参数
4. 测试对话

### 在 Dify 工作流中使用

1. 创建工作流应用
2. 添加 LLM 节点
3. 从下拉菜单中选择您的 AI 网关模型
4. 配置输入变量和输出处理
5. 运行和测试工作流

### 示例: 使用 JSON Schema 的结构化输出

```json
{
  "response_format": "json_schema",
  "json_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "number"},
      "email": {"type": "string"}
    },
    "required": ["name", "email"]
  }
}
```

插件会自动将 Schema 注入系统提示词以指导模型的响应。

### 示例: 使用思考模式(推理模型)

对于支持推理的模型(如 o1 或 DeepSeek R1):

1. 在模型配置中,将 `agent_though_support` 设置为 `supported` 或 `only_thinking_supported`
2. 在应用中启用"思考模式"参数
3. 模型将在最终答案前输出其推理过程

## 架构

### 认证流程

```
Dify 应用
      ↓
[AI 网关插件]
      ↓
[认证模块]
      ├── API Key → Authorization 头中的 Bearer token
      ├── JWT → 生成带有声明的签名令牌
      └── HMAC → 使用访问密钥 + 密钥签名请求
      ↓
[阿里云 AI 网关]
      ↓
[AI 模型服务]
```

## 故障排除

### 常见问题

1. **"认证失败"错误**
   - 验证您的凭证是否正确
   - 检查您的消费者在 AI 网关控制台中是否处于活动状态
   - 确保端点 URL 正确且可访问

2. **"模型未找到"错误**
   - 验证模型名称与 AI 网关配置完全匹配
   - 检查模型是否已在您的 API 配置中发布

3. **限流错误**
   - 检查您的消费者在 AI 网关中的限流设置
   - 考虑在应用中实现请求队列
   - 检查 Token 消耗限制

4. **超时错误**
   - 较长的响应可能超过超时限制
   - 考虑使用流式模式以获得更好的用户体验
   - 检查 AI 网关的超时配置

### 调试模式

启用调试日志:

```bash
# 运行前设置环境变量
export DIFY_PLUGIN_LOG_LEVEL=DEBUG
python -m main
```

## 安全最佳实践

1. **凭证管理**:
   - 切勿将凭证提交到版本控制
   - 使用环境变量或密钥管理系统
   - 定期轮换密钥

2. **HMAC 认证**:
   - 生产环境建议使用 HMAC
   - 保持密钥安全,切勿暴露
   - 监控签名验证失败情况

3. **AI 网关安全功能**:
   - 启用内容审核以过滤不当内容
   - 使用提示词注入检测防止攻击
   - 配置限流以防止滥用
   - 启用请求日志以进行审计

## 高级配置

### 自定义请求头

您可以通过设置 `extra_headers` 参数向 AI 网关传递自定义请求头:

```python
{
  "extra_headers": {
    "X-Custom-Header": "value"
  }
}
```

### 多模型策略

AI 网关支持模型路由。您可以配置多个模型,让 AI 网关根据以下条件路由请求:
- Token 消耗配额
- 模型可用性
- 响应时间要求
- 成本优化

## 贡献

欢迎贡献!请随时提交问题或拉取请求。

### 开发环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd dify-plugin-ai-gateway

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install pytest black flake8

# 运行代码检查
flake8 models/ provider/

# 格式化代码
black models/ provider/
```

## 许可证

本插件采用 MIT 许可证。详见 LICENSE 文件。

## 支持

- **Dify 文档**: https://docs.dify.ai
- **AI 网关文档**: https://help.aliyun.com/zh/api-gateway/ai-gateway

