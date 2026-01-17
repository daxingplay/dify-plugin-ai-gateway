import json
from typing import Mapping, Optional
from urllib.parse import urljoin

import requests

from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.rerank import RerankResult
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError
from dify_plugin.interfaces.model.openai_compatible.rerank import OAICompatRerankModel

from models.common.auth import get_api_path, prepare_auth_headers
from models.common.helpers import apply_display_name


class AiGatewayRerankModel(OAICompatRerankModel):
    """Model class for ai-gateway rerank model."""

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            self._invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of the United States?",
                docs=[
                    "Carson City is the capital city of the American state "
                    "of Nevada. At the 2010 United States "
                    "Census, Carson City had a population of 55,274.",
                    "The Commonwealth of the Northern Mariana Islands is a "
                    "group of islands in the Pacific Ocean that "
                    "are a political division controlled by the United "
                    "States. Its capital is Saipan.",
                    "Washington, D.C., formally the District of Columbia, "
                    "is the capital city and federal district of "
                    "the United States. It is located on the east bank of "
                    "the Potomac River.",
                ],
                score_threshold=0.8,
                top_n=3,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        apply_display_name(entity, credentials)

        return entity

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        """
        Invoke rerank model

        :param model: model name
        :param credentials: model credentials
        :param query: query
        :param docs: documents
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id
        :return: rerank result
        """
        endpoint_url = credentials.get("endpoint_url", "")
        if not endpoint_url:
            raise InvokeError("endpoint_url is required")

        # Handle vLLM protocol path
        # If user provides a base URL (e.g. https://api.example.com/v1), append /rerank
        # If user provides full path to rerank service, use it as is
        # However, standard vLLM is /v1/rerank or just /rerank relative to base
        # We use get_api_path logic but customized for rerank
        
        # Determine API path
        # vLLM standard is POST /v1/rerank
        api_suffix = "/rerank"
        api_path = get_api_path(credentials, api_suffix)
        full_url = urljoin(endpoint_url, api_path)
        if not full_url.endswith(api_suffix) and not "rerank" in full_url:
            # Fallback: if path resolution didn't add rerank and it's not in url, add it
             full_url = urljoin(full_url, "rerank")

        # Prepare payload (vLLM format)
        payload = {
            "model": credentials.get("endpoint_model_name") or model,
            "query": query,
            "documents": docs,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        
        # Note: vLLM might not support score_threshold in API standard params, 
        # but we can filter client side if needed. 
        # Some implementations might support it. We'll leave it out of payload 
        # to be safe unless we know it's supported, or filter results later.
        # Actually standard vLLM /v1/rerank arguments: model, query, documents, top_n, return_documents
        
        body_bytes = json.dumps(payload).encode("utf-8")

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        prepare_auth_headers(
            credentials=credentials,
            method="POST",
            path=api_path,
            body=body_bytes,
            extra_headers=headers,
        )

        try:
            response = requests.post(
                full_url,
                data=body_bytes,
                headers=headers,
                timeout=(10, 60),
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise InvokeError(f"Request failed: {e}")

        try:
            # Parse vLLM response
            # Expected format:
            # {
            #   "object": "list",
            #   "results": [
            #     {"index": 0, "relevance_score": 0.9},
            #     ...
            #   ]
            # }
            data = response.json()
            if "results" not in data:
                 raise InvokeError(f"Invalid response format: missing 'results' key. Response: {data}")
            
            results = data["results"]
            
            rerank_results = []
            for result in results:
                score = result.get("relevance_score")
                index = result.get("index")
                
                if score is None or index is None:
                    continue
                    
                if score_threshold is not None and score < score_threshold:
                    continue
                
                # Document object is not always returned by vLLM, so we rely on index mapping
                if 0 <= index < len(docs):
                    doc_text = docs[index]
                    rerank_results.append({
                        "index": index,
                        "score": score,
                        "document": {
                            "text": doc_text
                        }
                    })

            # Sort by score descending if not already (vLLM usually returns sorted)
            rerank_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply top_n limit again just in case (though vLLM handles it)
            if top_n is not None:
                rerank_results = rerank_results[:top_n]
                
            return RerankResult(
                usage=data.get("usage", {}),
                docs=rerank_results
            )

        except Exception as e:
             raise InvokeError(f"Failed to parse response: {e}")
