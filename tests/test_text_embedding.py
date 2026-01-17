from unittest.mock import MagicMock, patch

import pytest
from dify_plugin.entities.model import EmbeddingInputType
from models.text_embedding.text_embedding import AiGatewayTextEmbeddingModel


@pytest.fixture
def model():
    return AiGatewayTextEmbeddingModel(model_schemas=[])


@pytest.fixture
def credentials():
    return {
        "auth_method": "api_key",
        "api_key": "test_key",
        "endpoint_url": "https://api.example.com/v1",
        "endpoint_model_name": "test-model",
        "document_prefix": "doc:",
        "query_prefix": "query:",
    }


def test_get_prefix(model, credentials):
    assert model._get_prefix(credentials, EmbeddingInputType.DOCUMENT) == "doc:"
    assert model._get_prefix(credentials, EmbeddingInputType.QUERY) == "query:"


def test_add_prefix(model):
    texts = ["hello", "world"]
    assert model._add_prefix(texts, "doc:") == ["doc: hello", "doc: world"]
    assert model._add_prefix(texts, "") == texts


def test_invoke_embedding(model, credentials):
    with patch("dify_plugin.interfaces.model.openai_compatible.text_embedding.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"total_tokens": 10}
        }
        mock_post.return_value = mock_response

        result = model._invoke(
            "test-model",
            credentials,
            ["hello"],
            input_type=EmbeddingInputType.DOCUMENT
        )
        
        assert len(result.embeddings) == 1
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
