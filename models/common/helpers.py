"""Helper utilities for AI Gateway models."""
from collections.abc import Mapping
from typing import Union

from dify_plugin.entities.model import AIModelEntity, I18nObject


def apply_display_name(
    entity: AIModelEntity, credentials: Union[Mapping, dict]
) -> None:
    """
    Apply display name from credentials to entity label.

    :param entity: Model entity to update
    :param credentials: Model credentials
    """
    if "display_name" in credentials and credentials["display_name"] != "":
        entity.label = I18nObject(
            en_US=credentials["display_name"],
            zh_Hans=credentials["display_name"],
        )

