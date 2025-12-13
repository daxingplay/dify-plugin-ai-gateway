# AI Gateway Plugin for Dify

A Dify plugin that integrates with [Alibaba Cloud AI Gateway](https://www.alibabacloud.com/help/en/api-gateway/ai-gateway/product-overview/what-is-an-ai-gateway), enabling access to AI models through the gateway's protocol conversion, security protection, traffic management, and observability features.

## Overview

This plugin allows Dify to connect with Alibaba Cloud AI Gateway, which serves as a central hub between AI applications and model services. AI Gateway provides:

- **Protocol Conversion**: Standardized interfaces for accessing various AI models
- **Security Protection**: Content moderation, prompt injection detection, and sensitive data filtering
- **Traffic Management**: Rate limiting, load balancing, and intelligent routing
- **Unified Observability**: Monitoring, logging, and token usage tracking
- **Cost Optimization**: Request caching and token-based billing

## What is Alibaba Cloud AI Gateway?

[Alibaba Cloud AI Gateway](https://www.alibabacloud.com/help/en/api-gateway/ai-gateway/product-overview/what-is-an-ai-gateway) is a gateway service for AI application integration. It provides:

### Key Capabilities

1. **Multi-Model Support**: Unified interface for accessing models from different providers (OpenAI, Anthropic, Alibaba Qwen, etc.)
2. **Multi-Modal Support**: Handles text, image, audio, and video processing with various transmission protocols
3. **Security & Compliance**: Built-in AI security features including:
   - Content moderation and compliance checking
   - Prompt injection attack prevention
   - Sensitive data detection
   - Malicious file detection
   - Digital watermarking
4. **Traffic Governance**: 
   - Consumer-based rate limiting
   - Token consumption limits
   - Concurrent request controls
   - Intelligent load balancing
5. **Observability & Analytics**:
   - Real-time request/response monitoring
   - Token consumption statistics
   - Cache hit rate tracking
   - Model performance metrics

## Features

### Supported Model Types

- ✅ **Large Language Models (LLM)**: Chat and completion models
- ✅ **Rerank**: Document reranking models
- ✅ **Speech-to-Text**: Audio transcription models
- ✅ **Text Embedding**: Vector embedding models
- ✅ **Text-to-Speech**: Speech synthesis models

### Authentication Methods

This plugin supports three authentication methods for AI Gateway:

1. **API Key**: Simple bearer token authentication
2. **JWT (JSON Web Token)**: Standards-based authentication with JWKS support
3. **HMAC**: Request signing with access key and secret key

### Additional Features

- **Structured Output**: JSON schema validation for LLM responses
- **Thinking Mode**: Support for models with reasoning capabilities (o1, DeepSeek R1, etc.)
- **Custom Model Configuration**: Per-model display names and parameters
- **Streaming Support**: Real-time streaming responses for all compatible models

## Installation

### Prerequisites

- Dify instance (self-hosted or cloud)
- Alibaba Cloud account with AI Gateway enabled
- Python 3.11+ (for development)

### Installing the Plugin

1. **From Dify Marketplace** (Recommended):
   - Navigate to your Dify instance
   - Go to **Plugins** → **Marketplace**
   - Search for "AI Gateway"
   - Click **Install**

2. **Manual Installation** (Development):
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd dify-plugin-ai-gateway
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Package the plugin
   dify-plugin plugin package .
   
   # Upload the .difypkg file to your Dify instance
   ```

## Configuration

### Step 1: Create an AI Gateway Consumer

Before using this plugin, you need to create a consumer in Alibaba Cloud AI Gateway. Follow the [official guide to create a consumer](https://www.alibabacloud.com/help/en/api-gateway/ai-gateway/user-guide/creating-a-consumer).

A consumer represents an identity that can access your AI Gateway APIs and is used for authentication, authorization, and billing.

### Step 2: Configure Authentication

When adding a model in Dify, you'll need to provide the following credentials:

#### Common Settings

- **Endpoint URL**: Your AI Gateway endpoint (e.g., `https://your-gateway-cn-hangzhou.alicloudapi.com/v1`)
- **Model Name**: The model identifier in AI Gateway (e.g., `qwen-turbo`, `qwen-max`)
- **Display Name**: (Optional) Custom name shown in Dify UI

#### Authentication Method 1: API Key

```yaml
Authentication Method: api_key
API Key: your-api-key-from-ai-gateway
```

**When to use**: Simple scenarios where you control the API key distribution.

#### Authentication Method 2: JWT (JSON Web Token)

```yaml
Authentication Method: jwt
JWT JWKS: {"k":"your-base64-key","kty":"oct","alg":"HS256"}
JWT Algorithm: HS256 (or RS256, ES256, etc.)
JWT Consumer ID: your-consumer-id
JWT Expiration: 7200 (seconds)
JWT Issuer: (optional) your-issuer
JWT Header Name: Authorization
JWT Header Prefix: Bearer
```

**When to use**: Enterprise scenarios requiring standard token-based authentication with automatic expiration.

**JWKS Format**: The JWKS (JSON Web Key Set) should contain your signing key. For HMAC algorithms (HS256/384/512), use:
```json
{
  "k": "base64-encoded-secret",
  "kty": "oct",
  "alg": "HS256"
}
```

#### Authentication Method 3: HMAC Signature

```yaml
Authentication Method: hmac
HMAC Access Key: your-access-key
HMAC Secret Key: your-secret-key
```

**When to use**: Scenarios where request integrity verification is required. Each request is signed with a timestamp and content hash.

**How HMAC works**:
1. Plugin constructs a canonical string from the request (method, path, headers, body)
2. Signs the string with your secret key using SHA256
3. Includes the signature, access key, and timestamp in request headers
4. AI Gateway verifies the signature to authenticate the request

### Step 3: Configure Model Parameters

Depending on the model type, you can configure additional parameters:

**For LLM Models**:
- Temperature, Top P, Top K
- Max tokens
- Response format (text, json_object, json_schema)
- Thinking mode (for reasoning models)
- Stop sequences

**For Embedding Models**:
- Document prefix
- Query prefix
- Embedding dimensions

**For Speech Models**:
- Language
- Voice selection
- Audio format

## Usage Examples

### Using with Dify Chatbot

1. Create a new chatbot application in Dify
2. In the model settings, select your AI Gateway model
3. Configure the system prompt and parameters
4. Test the conversation

### Using with Dify Workflow

1. Create a workflow application
2. Add an LLM node
3. Select your AI Gateway model from the dropdown
4. Configure input variables and output processing
5. Run and test the workflow

### Example: Structured Output with JSON Schema

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

The plugin automatically injects the schema into the system prompt to guide the model's response.

### Example: Using Thinking Mode (Reasoning Models)

For models that support reasoning (like o1 or DeepSeek R1):

1. In model configuration, set `agent_though_support` to `supported` or `only_thinking_supported`
2. In your application, enable the "Thinking Mode" parameter
3. The model will output its reasoning process before the final answer

## Architecture

### Authentication Flow

```
Dify Application
      ↓
[AI Gateway Plugin]
      ↓
[Authentication Module]
      ├── API Key → Bearer token in Authorization header
      ├── JWT → Generate signed token with claims
      └── HMAC → Sign request with access key + secret
      ↓
[Alibaba Cloud AI Gateway]
      ↓
[AI Model Service]
```

## Troubleshooting

### Common Issues

1. **"Authentication failed" error**
   - Verify your credentials are correct
   - Check that your consumer is active in AI Gateway console
   - Ensure the endpoint URL is correct and accessible

2. **"Model not found" error**
   - Verify the model name matches exactly with AI Gateway configuration
   - Check that the model is published in your API configuration

3. **Rate limit errors**
   - Check your consumer's rate limit settings in AI Gateway
   - Consider implementing request queuing in your application
   - Review token consumption limits

4. **Timeout errors**
   - Long responses may exceed timeout limits
   - Consider using streaming mode for better user experience
   - Check AI Gateway's timeout configuration

### Debug Mode

To enable debug logging:

```bash
# Set environment variable before running
export DIFY_PLUGIN_LOG_LEVEL=DEBUG
python -m main
```

## Security Best Practices

1. **Credential Management**:
   - Never commit credentials to version control
   - Use environment variables or secret management systems
   - Rotate keys regularly

2. **HMAC Authentication**:
   - Prefer HMAC for production environments
   - Keep secret keys secure and never expose them
   - Monitor for signature verification failures

3. **AI Gateway Security Features**:
   - Enable content moderation to filter inappropriate content
   - Use prompt injection detection to prevent attacks
   - Configure rate limiting to prevent abuse
   - Enable request logging for audit trails

## Advanced Configuration

### Custom Headers

You can pass custom headers to AI Gateway by setting the `extra_headers` parameter:

```python
{
  "extra_headers": {
    "X-Custom-Header": "value"
  }
}
```

### Multi-Model Strategy

AI Gateway supports model routing. You can configure multiple models and let AI Gateway route requests based on:
- Token consumption quotas
- Model availability
- Response time requirements
- Cost optimization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd dify-plugin-ai-gateway

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Run linter
flake8 models/ provider/

# Format code
black models/ provider/
```

## License

This plugin is licensed under the MIT License. See LICENSE file for details.

## Support

- **Dify Documentation**: https://docs.dify.ai
- **AI Gateway Documentation**: https://www.alibabacloud.com/help/en/api-gateway/ai-gateway

