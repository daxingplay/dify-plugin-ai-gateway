"""Schema utilities for AI Gateway models."""

from dify_plugin.entities.model import (
    AIModelEntity,
    I18nObject,
    ParameterRule,
    ParameterType,
)


def add_auth_parameter_rules(entity: AIModelEntity) -> None:
    """
    Add authentication parameter rules to entity.

    :param entity: Model entity to add rules to
    """
    # Authentication selection (per model, no fallback)
    entity.parameter_rules.append(
        ParameterRule(
            name="auth_method",
            label=I18nObject(en_US="Authentication Method", zh_Hans="认证方式"),
            help=I18nObject(
                en_US=("Choose one auth method: API Key, JWT, or HMAC. " "Required."),
                zh_Hans="选择一种认证方式：API Key、JWT 或 HMAC。必填。",
            ),
            type=ParameterType.STRING,
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
            type=ParameterType.TEXT,
            required=False,
            show_on=[{"variable": "auth_method", "value": "jwt"}],
        )
    )
    entity.parameter_rules.append(
        ParameterRule(
            name="jwt_algorithm",
            label=I18nObject(en_US="JWT Algorithm", zh_Hans="JWT 算法"),
            type=ParameterType.STRING,
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
                    "Set the uid claim that matches the consumer ID " "in AI Gateway."
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
                en_US=("Expiration in seconds (max 604800 / 7 days). " "Default 7200."),
                zh_Hans="过期时间（秒），最大 604800/7 天，默认 7200。",
            ),
            type=ParameterType.INT,
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
            type=ParameterType.STRING,
            required=False,
            show_on=[{"variable": "auth_method", "value": "hmac"}],
        )
    )
