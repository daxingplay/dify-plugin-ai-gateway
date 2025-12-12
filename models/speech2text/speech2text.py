from typing import IO, Optional
from urllib.parse import urljoin

import requests
from dify_plugin.entities.model import AIModelEntity, FetchFrom, I18nObject, ModelType
from dify_plugin.errors.model import InvokeBadRequestError
from dify_plugin.interfaces.model.openai_compatible.speech2text import (
    OAICompatSpeech2TextModel,
)

from models.common.auth import build_hmac_headers, generate_jwt_token, get_api_path
from models.common.helpers import apply_display_name
from models.common.schema import add_auth_parameter_rules


class AiGatewaySpeech2TextModel(OAICompatSpeech2TextModel):
    """Model class for ai-gateway speech2text model."""

    def _invoke(
        self,
        model: str,
        credentials: dict,
        file: IO[bytes],
        user: Optional[str] = None,
    ) -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :param user: unique user id
        :return: text for given audio file
        """
        headers = {}

        # Prepare authentication headers
        auth_method = credentials.get("auth_method", "api_key")
        if auth_method == "api_key":
            api_key = credentials.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        elif auth_method == "jwt":
            token = generate_jwt_token(credentials)
            header_name = credentials.get("jwt_header_name") or "Authorization"
            header_prefix = credentials.get("jwt_header_prefix") or "Bearer"
            headers[header_name] = f"{header_prefix} {token}".strip()
        elif auth_method == "hmac":
            # For multipart/form-data, HMAC signing is complex
            # We'll use a simplified approach - sign with empty body
            # Note: This may need adjustment based on AI Gateway requirements
            api_path = get_api_path(credentials, "/audio/transcriptions")
            # Create a minimal body representation for signing
            # In practice, HMAC with multipart may require special handling
            body_bytes = b""  # Empty for multipart - may need adjustment
            access_key = credentials.get("hmac_access_key")
            secret_key = credentials.get("hmac_secret_key")
            if not access_key or not secret_key:
                raise ValueError(
                    "hmac_access_key and hmac_secret_key are required " "for HMAC auth"
                )
            hmac_headers = build_hmac_headers(
                method="POST",
                path=api_path,
                body=body_bytes,
                access_key=access_key,
                secret_key=secret_key,
                signature_headers=None,
            )
            headers.update(hmac_headers)

        endpoint_url = credentials.get("endpoint_url")
        if not endpoint_url:
            raise ValueError("endpoint_url is required")
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        endpoint_url = urljoin(endpoint_url, "audio/transcriptions")

        language = credentials.get("language", "en")
        prompt = credentials.get("initial_prompt", "convert the audio to text")
        payload = {
            "model": credentials.get("endpoint_model_name", model),
            "language": language,
            "prompt": prompt,
        }
        files = [("file", file)]
        response = requests.post(
            endpoint_url, headers=headers, data=payload, files=files
        )  # noqa: S113

        if response.status_code != 200:
            raise InvokeBadRequestError(response.text)
        response_data = response.json()
        return response_data["text"]

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        """
        used to define customizable model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.SPEECH2TEXT,
            model_properties={},
            parameter_rules=[],
        )

        apply_display_name(entity, credentials)
        add_auth_parameter_rules(entity)

        return entity
