from unittest.mock import MagicMock, patch

import pytest
from models.rerank.rerank import AiGatewayRerankModel


@pytest.fixture
def model():
    return AiGatewayRerankModel(model_schemas=[])


@pytest.fixture
def credentials():
    return {
        "auth_method": "api_key",
        "api_key": "test_key",
        "endpoint_url": "https://api.example.com/v1",
        "endpoint_model_name": "test-model",
    }


def test_invoke_rerank(model, credentials):
    with patch("models.rerank.rerank.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.1}
            ],
            "usage": {"total_tokens": 10}
        }
        mock_post.return_value = mock_response

        result = model._invoke(
            "test-model",
            credentials,
            query="test query",
            docs=["doc1", "doc2"],
            score_threshold=0.5
        )
        
        assert len(result.docs) == 1
        assert result.docs[0].score == 0.9
        assert result.docs[0].text == "doc1"
