import json
import re
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Dict, Generator, List, Optional, Union

from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    I18nObject,
    ModelFeature,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
)
from dify_plugin.interfaces.model.openai_compatible.llm import (
    OAICompatLargeLanguageModel,
)

from models.common.auth import get_api_path, prepare_auth_headers

logger = None


class AiGatewayLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    Model class for ai-gateway large language model.
    """

    # Pre-compiled regex for better performance
    _THINK_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)

    def get_customizable_model_schema(
        self, model: str, credentials: Union[Mapping, dict]
    ) -> AIModelEntity:
        """
        If your model supports fine-tuning, this method returns the schema
        of the base model but renamed to the fine-tuned model name.

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        entity = super().get_customizable_model_schema(model, credentials)

        structured_output_support = credentials.get(
            "structured_output_support", "not_supported"
        )
        if structured_output_support == "supported":
            entity.parameter_rules.append(
                ParameterRule(
                    name=DefaultParameterName.RESPONSE_FORMAT.value,
                    label=I18nObject(en_US="Response Format", zh_Hans="回复格式"),
                    help=I18nObject(
                        en_US=("Specifying the format that the model must " "output."),
                        zh_Hans="指定模型必须输出的回复格式。",
                    ),
                    type=ParameterType.STRING,
                    options=["text", "json_object", "json_schema"],
                    required=False,
                )
            )
            entity.parameter_rules.append(
                ParameterRule(
                    name="reasoning_format",
                    label=I18nObject(en_US="Reasoning Format", zh_Hans="推理格式"),
                    help=I18nObject(
                        en_US=(
                            "Specifying the format that the model must "
                            "output reasoning."
                        ),
                        zh_Hans="指定模型必须输出的推理格式。",
                    ),
                    type=ParameterType.STRING,
                    options=["none", "auto", "deepseek", "deepseek-legacy"],
                    required=False,
                )
            )
            entity.parameter_rules.append(
                ParameterRule(
                    name=DefaultParameterName.JSON_SCHEMA.value,
                    use_template=DefaultParameterName.JSON_SCHEMA.value,
                )
            )

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(
                en_US=credentials["display_name"],
                zh_Hans=credentials["display_name"],
            )

        # Authentication selection (per model, no fallback)
        entity.parameter_rules.append(
            ParameterRule(
                name="auth_method",
                label=I18nObject(en_US="Authentication Method", zh_Hans="认证方式"),
                help=I18nObject(
                    en_US=(
                        "Choose one auth method: API Key, JWT, or HMAC. " "Required."
                    ),
                    zh_Hans="选择一种认证方式：API Key、JWT 或 HMAC。必填。",
                ),
                type=ParameterType.SELECT,
                options=["api_key", "jwt", "hmac"],
                required=True,
            )
        )

        # API Key: keep existing api_key field
        # (conditional requirement handled in UI)

        # JWT fields (shown when auth_method = jwt)
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_jwks",
                label=I18nObject(en_US="JWT JWKS", zh_Hans="JWT JWKS"),
                help=I18nObject(
                    en_US=(
                        "JWKS JSON used to sign JWT tokens "
                        '(e.g. {"k":"...","kty":"oct",'
                        '"alg":"HS256"}).'
                    ),
                    zh_Hans=(
                        "用于签发JWT的JWKS JSON（如 "
                        '{"k":"...","kty":"oct","alg":"HS256"}）。'
                    ),
                ),
                type=ParameterType.TEXT_AREA,
                required=False,
                show_on=[{"variable": "auth_method", "value": "jwt"}],
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_algorithm",
                label=I18nObject(en_US="JWT Algorithm", zh_Hans="JWT 算法"),
                type=ParameterType.SELECT,
                options=[
                    "HS256",
                    "HS384",
                    "HS512",
                    "RS256",
                    "RS384",
                    "RS512",
                    "ES256",
                    "ES384",
                    "ES512",
                    "PS256",
                    "PS384",
                    "PS512",
                    "EdDSA",
                ],
                required=False,
                show_on=[{"variable": "auth_method", "value": "jwt"}],
                default="HS256",
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_consumer_id",
                label=I18nObject(
                    en_US="JWT Consumer ID (uid)", zh_Hans="JWT 消费者标识(uid)"
                ),
                help=I18nObject(
                    en_US=(
                        "Set the uid claim that matches the consumer ID "
                        "in AI Gateway."
                    ),
                    zh_Hans="设置与 AI Gateway 消费者 ID 匹配的 uid 声明。",
                ),
                type=ParameterType.STRING,
                required=False,
                show_on=[{"variable": "auth_method", "value": "jwt"}],
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_expiration",
                label=I18nObject(
                    en_US="JWT Expiration (seconds)", zh_Hans="JWT 过期时间(秒)"
                ),
                help=I18nObject(
                    en_US=(
                        "Expiration in seconds (max 604800 / 7 days). " "Default 7200."
                    ),
                    zh_Hans="过期时间（秒），最大 604800/7 天，默认 7200。",
                ),
                type=ParameterType.NUMBER,
                required=False,
                default=7200,
                show_on=[{"variable": "auth_method", "value": "jwt"}],
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_issuer",
                label=I18nObject(en_US="JWT Issuer (iss)", zh_Hans="JWT 签发者(iss)"),
                type=ParameterType.STRING,
                required=False,
                show_on=[{"variable": "auth_method", "value": "jwt"}],
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_header_name",
                label=I18nObject(en_US="JWT Header Name", zh_Hans="JWT 头名称"),
                help=I18nObject(
                    en_US="Header key for JWT token (default Authorization).",
                    zh_Hans="JWT 令牌使用的请求头键（默认 Authorization）。",
                ),
                type=ParameterType.STRING,
                required=False,
                default="Authorization",
                show_on=[{"variable": "auth_method", "value": "jwt"}],
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="jwt_header_prefix",
                label=I18nObject(en_US="JWT Header Prefix", zh_Hans="JWT 头前缀"),
                help=I18nObject(
                    en_US="Header prefix for JWT token (default Bearer).",
                    zh_Hans="JWT 令牌使用的头前缀（默认 Bearer）。",
                ),
                type=ParameterType.STRING,
                required=False,
                default="Bearer",
                show_on=[{"variable": "auth_method", "value": "jwt"}],
            )
        )

        # HMAC fields (shown when auth_method = hmac)
        entity.parameter_rules.append(
            ParameterRule(
                name="hmac_access_key",
                label=I18nObject(en_US="HMAC Access Key", zh_Hans="HMAC Access Key"),
                type=ParameterType.STRING,
                required=False,
                show_on=[{"variable": "auth_method", "value": "hmac"}],
            )
        )
        entity.parameter_rules.append(
            ParameterRule(
                name="hmac_secret_key",
                label=I18nObject(en_US="HMAC Secret Key", zh_Hans="HMAC Secret Key"),
                type=ParameterType.SECRET_INPUT,
                required=False,
                show_on=[{"variable": "auth_method", "value": "hmac"}],
            )
        )

        # Configure thinking mode parameter based on model support
        agent_though_support = credentials.get("agent_though_support", "not_supported")

        # Add AGENT_THOUGHT feature if thinking mode is supported
        # (either mode)
        if (
            agent_though_support in ["supported", "only_thinking_supported"]
            and ModelFeature.AGENT_THOUGHT not in entity.features
        ):
            entity.features.append(ModelFeature.AGENT_THOUGHT)

        # Only add the enable_thinking parameter if the model supports
        # both modes. If only_thinking_supported, the parameter is not
        # needed (forced behavior)
        if agent_though_support == "supported":
            entity.parameter_rules.append(
                ParameterRule(
                    name="enable_thinking",
                    label=I18nObject(en_US="Thinking mode", zh_Hans="思考模式"),
                    help=I18nObject(
                        en_US=(
                            "Whether to enable thinking mode, applicable "
                            "to various thinking mode models deployed on "
                            "reasoning frameworks such as vLLM and SGLang, "
                            "for example Qwen3."
                        ),
                        zh_Hans=(
                            "是否开启思考模式，适用于vLLM和SGLang等推理框架"
                            "部署的多种思考模式模型，例如Qwen3。"
                        ),
                    ),
                    type=ParameterType.BOOLEAN,
                    required=False,
                )
            )

        return entity

    @classmethod
    def _drop_analyze_channel(cls, prompt_messages: List[PromptMessage]) -> None:
        """
        Remove thinking content from assistant messages for better performance.

        Uses early exit and pre-compiled regex to minimize overhead.
        Args:
            prompt_messages:

        Returns:

        """
        for p in prompt_messages:
            # Early exit conditions
            if not isinstance(p, AssistantPromptMessage):
                continue
            if not isinstance(p.content, str):
                continue
            # Quick check to avoid regex if not needed
            if not p.content.startswith("<think>"):
                continue

            # Only perform regex substitution when necessary
            new_content = cls._THINK_PATTERN.sub("", p.content, count=1)
            # Only update if changed
            if new_content != p.content:
                p.content = new_content

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        # Prepare authentication headers using common auth module
        extra_headers: Dict[str, str] = model_parameters.get("extra_headers", {}) or {}

        # For HMAC, we need to build the request body first to sign it
        auth_method = credentials.get("auth_method")
        if auth_method == "hmac":
            # Build an approximate OpenAI-compatible request body for signing
            # Note: this mirrors the expected body shape for chat/completions
            payload: Dict[str, Any] = {
                "model": credentials.get("endpoint_model_name") or model,
            }
            # Attach parameters except internal helpers
            for k, v in model_parameters.items():
                if k == "extra_headers":
                    continue
                payload[k] = v

            # Messages serialization (best effort)
            if prompt_messages:
                messages: List[Dict[str, Any]] = []
                for msg in prompt_messages:
                    if hasattr(msg, "role"):
                        role = msg.role
                    else:
                        role = None
                    content = getattr(msg, "content", None)
                    messages.append({"role": role, "content": content})
                payload["messages"] = messages

            body_bytes = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode()

            # Derive path from endpoint_url and mode
            mode = credentials.get("mode", "chat")
            api_path = "/completions" if mode == "completion" else "/chat/completions"
            path = get_api_path(credentials, api_path)

            # Prepare auth headers using common module
            prepare_auth_headers(
                credentials=credentials,
                method="POST",
                path=path,
                body=body_bytes,
                extra_headers=extra_headers,
            )
        else:
            # For JWT and API key, we can use prepare_auth_headers with empty body
            # The path doesn't matter for JWT/API key, but we'll use a default
            mode = credentials.get("mode", "chat")
            api_path = "/completions" if mode == "completion" else "/chat/completions"
            path = get_api_path(credentials, api_path)
            # Empty body for JWT/API key (not used for signing)
            body_bytes = b""
            prepare_auth_headers(
                credentials=credentials,
                method="POST",
                path=path,
                body=body_bytes,
                extra_headers=extra_headers,
            )

        if extra_headers:
            model_parameters["extra_headers"] = extra_headers

        # Compatibility adapter for Dify's 'json_schema' structured output
        # mode. The base class does not natively handle the 'json_schema'
        # parameter. This block translates it into a standard
        # OpenAI-compatible request by:
        # 1. Injecting the JSON schema directly into the system prompt to
        #    guide the model.
        # This ensures models like gpt-4o produce the correct structured
        # output.
        if model_parameters.get("response_format") == "json_schema":
            # Use .get() instead of .pop() for safety
            json_schema_str = model_parameters.get("json_schema")

            if json_schema_str:
                structured_output_prompt = (
                    "Your response must be a JSON object that validates "
                    "against the following JSON schema, and nothing else.\n"
                    f"JSON Schema: ```json\n{json_schema_str}\n```"
                )

                existing_system_prompt = next(
                    (p for p in prompt_messages if p.role == PromptMessageRole.SYSTEM),
                    None,
                )
                if existing_system_prompt:
                    existing_system_prompt.content = (
                        structured_output_prompt
                        + "\n\n"
                        + existing_system_prompt.content
                    )
                else:
                    prompt_messages.insert(
                        0,
                        SystemPromptMessage(content=structured_output_prompt),
                    )

        # Handle thinking mode based on model support configuration
        agent_though_support = credentials.get("agent_though_support", "not_supported")
        enable_thinking_value = None
        if agent_though_support == "only_thinking_supported":
            # Force enable thinking mode
            enable_thinking_value = True
        elif agent_though_support == "not_supported":
            # Force disable thinking mode
            enable_thinking_value = False
        else:
            # Both modes supported - use user's preference
            user_enable_thinking = model_parameters.pop("enable_thinking", None)
            if user_enable_thinking is not None:
                enable_thinking_value = bool(user_enable_thinking)

        if enable_thinking_value is not None:
            model_parameters.setdefault("chat_template_kwargs", {})[
                "enable_thinking"
            ] = enable_thinking_value
            model_parameters.setdefault("chat_template_kwargs", {})[
                "thinking"
            ] = enable_thinking_value

        # Remove thinking content from assistant messages for better
        # performance.
        with suppress(Exception):
            self._drop_analyze_channel(prompt_messages)

        return super()._invoke(
            model,
            credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )
