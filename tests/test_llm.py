from unittest.mock import MagicMock, patch

import pytest
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.message import UserPromptMessage

from models.llm.llm import AiGatewayLargeLanguageModel


@pytest.fixture
def model():
    return AiGatewayLargeLanguageModel(model_schemas=[])


@pytest.fixture
def credentials():
    return {
        "auth_method": "api_key",
        "api_key": "test_key",
        "endpoint_url": "https://api.example.com/v1",
        "endpoint_model_name": "test-model",
        "mode": "chat",
    }


@pytest.fixture
def qwen_credentials():
    """Credentials for Qwen (DashScope) models with thinking mode."""
    return {
        "auth_method": "api_key",
        "api_key": "test_key",
        "endpoint_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "endpoint_model_name": "qwen-plus",
        "mode": "chat",
        "agent_though_support": "supported",
        "thinking_mode_provider": "qwen",
    }


def test_validate_credentials_api_key(model, credentials):
    with patch("dify_plugin.interfaces.model.openai_compatible.llm.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        }
        mock_post.return_value = mock_response

        # Should not raise exception
        model.validate_credentials("test-model", credentials)


def test_validate_credentials_custom_auth(model, credentials):
    credentials["auth_method"] = "jwt"
    credentials["jwt_header_name"] = "Authorization"
    credentials["jwt_header_prefix"] = "Bearer"
    
    with patch("models.llm.llm.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        with patch("models.llm.llm.prepare_auth_headers") as mock_prep:
            model.validate_credentials("test-model", credentials)
            mock_prep.assert_called_once()


def test_invoke_llm(model, credentials):
    with patch("models.llm.llm.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello", "role": "assistant"}}]
        }
        mock_post.return_value = mock_response

        messages = [UserPromptMessage(content="Hi")]
        response = model._invoke(
            "test-model", 
            credentials, 
            messages, 
            {"temperature": 0.7},
            stream=False
        )
        
        assert response.message.content == "Hello"


class TestQwenThinkingMode:
    """Test cases for Qwen (DashScope) thinking mode parameters."""

    def test_qwen_thinking_mode_enabled(self, model, qwen_credentials):
        """Test Qwen thinking mode with enable_thinking=True."""
        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "The answer is 42.",
                        "role": "assistant",
                        "reasoning_content": "Let me think..."
                    }
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="What is the answer?")]
            model_parameters = {
                "temperature": 0.7,
                "enable_thinking": True,
            }

            model._invoke(
                "qwen-plus",
                qwen_credentials,
                messages,
                model_parameters,
                stream=False
            )

            # Verify enable_thinking is passed directly in request body
            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            assert request_json.get("enable_thinking") is True
            # chat_template_kwargs should NOT be used for Qwen
            assert "chat_template_kwargs" not in request_json

    def test_qwen_thinking_mode_with_budget(self, model, qwen_credentials):
        """Test Qwen thinking mode with thinking_budget parameter."""
        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Result",
                        "role": "assistant"
                    }
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="Think carefully")]
            model_parameters = {
                "enable_thinking": True,
                "thinking_budget": 10000,
            }

            model._invoke(
                "qwen-plus",
                qwen_credentials,
                messages,
                model_parameters,
                stream=False
            )

            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            assert request_json.get("enable_thinking") is True
            assert request_json.get("thinking_budget") == 10000

    def test_qwen_thinking_mode_disabled_no_budget(
        self, model, qwen_credentials
    ):
        """Test that thinking_budget is not sent when thinking is disabled."""
        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Hello", "role": "assistant"}
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="Hi")]
            model_parameters = {
                "enable_thinking": False,
                "thinking_budget": 5000,  # Should be ignored
            }

            model._invoke(
                "qwen-plus",
                qwen_credentials,
                messages,
                model_parameters,
                stream=False
            )

            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            assert request_json.get("enable_thinking") is False
            # thinking_budget should NOT be sent when thinking is disabled
            assert "thinking_budget" not in request_json

    def test_vllm_thinking_mode_uses_chat_template_kwargs(
        self, model, credentials
    ):
        """Test vLLM/SGLang provider uses chat_template_kwargs."""
        vllm_credentials = {
            **credentials,
            "agent_though_support": "supported",
            "thinking_mode_provider": "vllm",
        }

        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Result", "role": "assistant"}
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="Think")]
            model_parameters = {
                "enable_thinking": True,
            }

            model._invoke(
                "test-model",
                vllm_credentials,
                messages,
                model_parameters,
                stream=False
            )

            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            # vLLM should use chat_template_kwargs
            assert "chat_template_kwargs" in request_json
            assert request_json["chat_template_kwargs"]["enable_thinking"]
            assert request_json["chat_template_kwargs"]["thinking"]
            # Direct enable_thinking should NOT be in body
            assert "enable_thinking" not in request_json

    def test_qwen_schema_includes_thinking_budget(self, model, qwen_credentials):
        """Test that Qwen model schema includes thinking_budget parameter."""
        entity = model.get_customizable_model_schema(
            "qwen-plus", qwen_credentials
        )

        param_names = [rule.name for rule in entity.parameter_rules]
        assert "enable_thinking" in param_names
        assert "thinking_budget" in param_names

        # Verify thinking_budget rule properties
        thinking_budget_rule = next(
            r for r in entity.parameter_rules if r.name == "thinking_budget"
        )
        assert thinking_budget_rule.min == 1
        assert thinking_budget_rule.max == 38000

    def test_vllm_schema_excludes_thinking_budget(self, model, credentials):
        """Test that vLLM model schema excludes thinking_budget parameter."""
        vllm_credentials = {
            **credentials,
            "agent_though_support": "supported",
            "thinking_mode_provider": "vllm",
        }

        entity = model.get_customizable_model_schema(
            "test-model", vllm_credentials
        )

        param_names = [rule.name for rule in entity.parameter_rules]
        assert "enable_thinking" in param_names
        # thinking_budget should NOT be available for vLLM
        assert "thinking_budget" not in param_names

    def test_qwen_only_thinking_mode_forced(self, model):
        """Test Qwen model with only_thinking_supported forces enable_thinking."""
        qwen_only_thinking_creds = {
            "auth_method": "api_key",
            "api_key": "test_key",
            "endpoint_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "endpoint_model_name": "qwen3-235b-a22b",
            "mode": "chat",
            "agent_though_support": "only_thinking_supported",
            "thinking_mode_provider": "qwen",
        }

        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Thinking...", "role": "assistant"}
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="Solve this")]
            # No enable_thinking in parameters - should be forced to True
            model_parameters = {"temperature": 0.7}

            model._invoke(
                "qwen3-235b-a22b",
                qwen_only_thinking_creds,
                messages,
                model_parameters,
                stream=False
            )

            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            # Should be forced to True
            assert request_json.get("enable_thinking") is True

    def test_qwen_agent_thought_feature(self, model, qwen_credentials):
        """Test that Qwen model with thinking support has AGENT_THOUGHT feature."""
        entity = model.get_customizable_model_schema(
            "qwen-plus", qwen_credentials
        )
        assert ModelFeature.AGENT_THOUGHT in entity.features

    def test_qwen_default_thinking_budget_from_credentials(self, model):
        """Test that default_thinking_budget from credentials is used."""
        qwen_creds_with_default = {
            "auth_method": "api_key",
            "api_key": "test_key",
            "endpoint_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "endpoint_model_name": "qwen-plus",
            "mode": "chat",
            "agent_though_support": "supported",
            "thinking_mode_provider": "qwen",
            "default_thinking_budget": "15000",
        }

        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Result", "role": "assistant"}
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="Think")]
            # No thinking_budget in parameters - should use credential default
            model_parameters = {"enable_thinking": True}

            model._invoke(
                "qwen-plus",
                qwen_creds_with_default,
                messages,
                model_parameters,
                stream=False
            )

            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            assert request_json.get("enable_thinking") is True
            # Should use default from credentials
            assert request_json.get("thinking_budget") == 15000

    def test_qwen_per_request_budget_overrides_default(self, model):
        """Test that per-request thinking_budget overrides credential default."""
        qwen_creds_with_default = {
            "auth_method": "api_key",
            "api_key": "test_key",
            "endpoint_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "endpoint_model_name": "qwen-plus",
            "mode": "chat",
            "agent_though_support": "supported",
            "thinking_mode_provider": "qwen",
            "default_thinking_budget": "15000",
        }

        with patch("models.llm.llm.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"content": "Result", "role": "assistant"}
                }]
            }
            mock_post.return_value = mock_response

            messages = [UserPromptMessage(content="Think")]
            # Per-request budget should override default
            model_parameters = {
                "enable_thinking": True,
                "thinking_budget": 5000,
            }

            model._invoke(
                "qwen-plus",
                qwen_creds_with_default,
                messages,
                model_parameters,
                stream=False
            )

            call_args = mock_post.call_args
            request_json = call_args.kwargs.get("json", {})
            # Per-request value should override credential default
            assert request_json.get("thinking_budget") == 5000


class TestModelFeatures:
    """Test cases for model feature flags."""

    def test_vision_support_adds_feature(self, model, credentials):
        """Test VISION feature added when vision_support='support'."""
        vision_credentials = {
            **credentials,
            "vision_support": "support",
        }

        entity = model.get_customizable_model_schema(
            "test-model", vision_credentials
        )
        assert ModelFeature.VISION in entity.features

    def test_vision_support_not_added_when_disabled(
        self, model, credentials
    ):
        """Test VISION NOT added when vision_support='no_support'."""
        no_vision_credentials = {
            **credentials,
            "vision_support": "no_support",
        }

        entity = model.get_customizable_model_schema(
            "test-model", no_vision_credentials
        )
        assert ModelFeature.VISION not in entity.features

    def test_stream_function_calling_adds_feature(
        self, model, credentials
    ):
        """Test STREAM_TOOL_CALL added when stream_function_calling='supported'."""
        stream_tool_credentials = {
            **credentials,
            "stream_function_calling": "supported",
        }

        entity = model.get_customizable_model_schema(
            "test-model", stream_tool_credentials
        )
        assert ModelFeature.STREAM_TOOL_CALL in entity.features

    def test_stream_function_calling_not_added_when_disabled(
        self, model, credentials
    ):
        """Test STREAM_TOOL_CALL NOT added when stream_function_calling='not_supported'."""
        no_stream_tool_credentials = {
            **credentials,
            "stream_function_calling": "not_supported",
        }

        entity = model.get_customizable_model_schema(
            "test-model", no_stream_tool_credentials
        )
        assert ModelFeature.STREAM_TOOL_CALL not in entity.features

    def test_features_not_added_when_disabled(self, model, credentials):
        """Test that features are NOT added when credentials are disabled."""
        disabled_credentials = {
            **credentials,
            "vision_support": "no_support",
            "stream_function_calling": "not_supported",
            "agent_though_support": "not_supported",
        }

        entity = model.get_customizable_model_schema(
            "test-model", disabled_credentials
        )
        assert ModelFeature.VISION not in entity.features
        assert ModelFeature.STREAM_TOOL_CALL not in entity.features
        assert ModelFeature.AGENT_THOUGHT not in entity.features

    def test_multiple_features_can_be_enabled(self, model, credentials):
        """Test that multiple features can be enabled simultaneously."""
        all_features_credentials = {
            **credentials,
            "vision_support": "support",
            "stream_function_calling": "supported",
            "agent_though_support": "supported",
        }

        entity = model.get_customizable_model_schema(
            "test-model", all_features_credentials
        )
        assert ModelFeature.VISION in entity.features
        assert ModelFeature.STREAM_TOOL_CALL in entity.features
        assert ModelFeature.AGENT_THOUGHT in entity.features
