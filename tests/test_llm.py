from unittest.mock import MagicMock, patch

import pytest
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
