import json
import logging
import re
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Dict, Generator, List, Optional, Union
from urllib.parse import urljoin

import requests
from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.entities.model import (AIModelEntity, DefaultParameterName,
                                        I18nObject, ModelFeature,
                                        ParameterRule, ParameterType)
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import (AssistantPromptMessage,
                                                PromptMessage,
                                                PromptMessageRole,
                                                PromptMessageTool,
                                                SystemPromptMessage)
from dify_plugin.errors.model import (CredentialsValidateFailedError,
                                      InvokeError)
from dify_plugin.interfaces.model.openai_compatible.llm import \
    OAICompatLargeLanguageModel
from pydantic import TypeAdapter

from models.common.auth import get_api_path, prepare_auth_headers

# Setup logger with Dify plugin handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(plugin_logger_handler)


class AiGatewayLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    Model class for ai-gateway large language model.
    """

    # Pre-compiled regex for better performance
    _THINK_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        logger.info(f"Validating credentials for model: {model}")
        auth_method = credentials.get("auth_method", "api_key")
        logger.debug(f"Auth method: {auth_method}")

        # For api_key auth, delegate to base class
        if auth_method == "api_key":
            try:
                super().validate_credentials(model, credentials)
                logger.info(
                    f"Credentials validated successfully for model: {model}"
                )
                return
            except UnboundLocalError as e:
                if "response" in str(e):
                    logger.error(
                        f"Credentials validation failed: UnboundLocalError - {e}"
                    )
                    raise CredentialsValidateFailedError(
                        "Credentials validation failed: authentication setup error"
                    ) from e
                raise
            except Exception as ex:
                exc_type = type(ex).__name__
                logger.error(
                    f"Credentials validation failed: {exc_type} - {ex}"
                )
                raise CredentialsValidateFailedError(str(ex)) from ex

        # For JWT and HMAC, make our own validation request
        # since the base class doesn't support these auth methods
        try:
            self._validate_credentials_with_custom_auth(model, credentials)
            logger.info(
                f"Credentials validated successfully for model: {model}"
            )
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            exc_type = type(ex).__name__
            logger.error(
                f"Credentials validation failed: {exc_type} - {ex}"
            )
            raise CredentialsValidateFailedError(str(ex)) from ex

    def _validate_credentials_with_custom_auth(
        self, model: str, credentials: dict
    ) -> None:
        """
        Validate credentials using custom HTTP request for JWT/HMAC auth.

        :param model: model name
        :param credentials: model credentials
        :raises CredentialsValidateFailedError: if validation fails
        """
        auth_method = credentials.get("auth_method")
        mode = credentials.get("mode", "chat")
        logger.debug(
            f"Custom auth validation: auth_method={auth_method}, mode={mode}"
        )

        # Build endpoint URL
        endpoint_url = credentials.get("endpoint_url", "")
        if not endpoint_url:
            raise CredentialsValidateFailedError("endpoint_url is required")
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"

        # Build request payload
        if mode == "completion":
            api_suffix = "completions"
            payload: Dict[str, Any] = {
                "model": credentials.get("endpoint_model_name") or model,
                "prompt": "ping",
                "max_tokens": 5,
            }
        else:
            api_suffix = "chat/completions"
            payload = {
                "model": credentials.get("endpoint_model_name") or model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            }

        full_url = urljoin(endpoint_url, api_suffix)
        api_path = get_api_path(credentials, f"/{api_suffix}")

        # Serialize payload for signing (HMAC) and sending
        body_bytes = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode()
        logger.debug(f"Validation request URL: {full_url}")
        logger.debug(f"Validation payload size: {len(body_bytes)} bytes")

        # Prepare auth headers first
        extra_headers: Dict[str, str] = {}
        prepare_auth_headers(
            credentials=credentials,
            method="POST",
            path=api_path,
            body=body_bytes,
            extra_headers=extra_headers,
        )

        # For HMAC, build_hmac_headers already provides content-type and accept
        # So we start with extra_headers to avoid duplicates
        if auth_method == "hmac":
            headers = extra_headers
        else:
            # For JWT, set base headers then merge
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            headers.update(extra_headers)
            # For JWT, the token is set in credentials["api_key"] by prepare_auth_headers
            api_key = credentials.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        logger.debug(f"Validation request headers: {list(headers.keys())}")

        # Make the validation request
        try:
            response = requests.post(
                full_url,
                headers=headers,
                data=body_bytes,
                timeout=(10, 30),
            )
        except requests.RequestException as ex:
            logger.error(f"Validation request failed: {ex}")
            raise CredentialsValidateFailedError(
                f"Connection failed: {ex}"
            ) from ex

        if response.status_code != 200:
            error_msg = (
                f"Validation failed with status {response.status_code}: "
                f"{response.text[:500]}"
            )
            logger.error(error_msg)
            raise CredentialsValidateFailedError(error_msg)

        logger.debug(f"Validation response status: {response.status_code}")

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
        extra_headers: Dict[str, str] = (
            model_parameters.get("extra_headers", {}) or {}
        )

        # For HMAC, we need to build the request body first to sign it
        auth_method = credentials.get("auth_method")
        logger.info(
            f"Invoking LLM model: {model}, auth_method: {auth_method}, "
            f"stream: {stream}"
        )
        logger.debug(f"Model parameters: {list(model_parameters.keys())}")

        if auth_method == "hmac":
            logger.debug("Building request body for HMAC signature")
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
                msg_count = len(messages)
                logger.debug(
                    f"Serialized {msg_count} messages for HMAC signing"
                )

            body_bytes = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode()
            body_len = len(body_bytes)
            logger.debug(f"Request body size for HMAC: {body_len} bytes")

            # Derive path from endpoint_url and mode
            mode = credentials.get("mode", "chat")
            api_path = (
                "/completions" if mode == "completion" else "/chat/completions"
            )
            path = get_api_path(credentials, api_path)
            logger.debug(f"API path for HMAC: {path}")

            # Prepare auth headers using common module
            prepare_auth_headers(
                credentials=credentials,
                method="POST",
                path=path,
                body=body_bytes,
                extra_headers=extra_headers,
            )
        else:
            # For JWT and API key, we can use prepare_auth_headers with
            # empty body. The path doesn't matter for JWT/API key, but
            # we'll use a default
            logger.debug(f"Preparing auth headers for {auth_method}")
            mode = credentials.get("mode", "chat")
            api_path = (
                "/completions" if mode == "completion" else "/chat/completions"
            )
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
            logger.debug(
                f"Extra headers prepared: {list(extra_headers.keys())}"
            )

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

    def _generate(
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
        Override _generate to avoid DifyPluginEnv.MAX_REQUEST_TIMEOUT issue.

        The base class tries to access DifyPluginEnv.MAX_REQUEST_TIMEOUT as
        a class attribute, but in dify_plugin 0.6.2+ it's only an instance
        attribute. We use a hardcoded timeout value instead.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8",
        }
        extra_headers = credentials.get("extra_headers")
        if extra_headers is not None:
            headers = {
                **headers,
                **extra_headers,
            }

        api_key = credentials.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = credentials["endpoint_url"]
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"

        response_format = model_parameters.get("response_format")
        if response_format:
            if response_format == "json_schema":
                json_schema = model_parameters.get("json_schema")
                if not json_schema:
                    raise ValueError(
                        "Must define JSON Schema when response format is " "json_schema"
                    )
                try:
                    schema = TypeAdapter(dict[str, Any]).validate_json(json_schema)
                except Exception as exc:
                    raise ValueError(
                        f"not correct json_schema format: {json_schema}"
                    ) from exc
                model_parameters.pop("json_schema")
                model_parameters["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema,
                }
            else:
                model_parameters["response_format"] = {"type": response_format}
        elif "json_schema" in model_parameters:
            del model_parameters["json_schema"]

        data = {
            "model": credentials.get("endpoint_model_name", model),
            "stream": stream,
            **model_parameters,
        }

        completion_type = LLMMode.value_of(credentials["mode"])

        if completion_type is LLMMode.CHAT:
            endpoint_url = urljoin(endpoint_url, "chat/completions")
            data["messages"] = [
                self._convert_prompt_message_to_dict(m, credentials)
                for m in prompt_messages
            ]
        elif completion_type is LLMMode.COMPLETION:
            endpoint_url = urljoin(endpoint_url, "completions")
            data["prompt"] = prompt_messages[0].content
        else:
            raise ValueError("Unsupported completion type for model configuration.")

        # annotate tools with names, descriptions, etc.
        from dify_plugin.entities.model.message import PromptMessageFunction

        function_calling_type = credentials.get("function_calling_type", "no_call")
        formatted_tools = []
        if tools:
            if function_calling_type == "function_call":
                data["functions"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            elif function_calling_type == "tool_call":
                data["tool_choice"] = "auto"

                for tool in tools:
                    formatted_tools.append(
                        PromptMessageFunction(function=tool).model_dump()
                    )

                data["tools"] = formatted_tools

        if stop:
            data["stop"] = stop

        if user:
            data["user"] = user

        # Use hardcoded timeout instead of DifyPluginEnv.MAX_REQUEST_TIMEOUT
        # which is not available as a class attribute in dify_plugin 0.6.2+
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=data,
            timeout=(10, 120),  # hardcoded: (connection, read) timeout
            stream=stream,
        )

        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = "utf-8"

        if response.status_code != 200:
            raise InvokeError(
                f"API request failed with status code "
                f"{response.status_code}: {response.text}"
            )

        if stream:
            return self._handle_generate_stream_response(
                model, credentials, response, prompt_messages
            )

        return self._handle_generate_response(
            model, credentials, response, prompt_messages
        )
