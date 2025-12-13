from typing import Mapping

from dify_plugin.entities.model import AIModelEntity
from dify_plugin.interfaces.model.openai_compatible.tts import OAICompatText2SpeechModel

from models.common.helpers import apply_display_name


class AiGatewayText2SpeechModel(OAICompatText2SpeechModel):
    """Model class for ai-gateway text2speech model."""

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        entity = super().get_customizable_model_schema(model, credentials)

        apply_display_name(entity, credentials)

        return entity
