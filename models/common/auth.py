"""Authentication utilities for AI Gateway models."""

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import jwt
from dify_plugin.config.logger_format import plugin_logger_handler

# Setup logger with Dify plugin handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(plugin_logger_handler)


def parse_jwks(credentials: dict) -> Dict[str, Any]:
    """Parse JWKS from credentials."""
    logger.debug("Parsing JWKS from credentials")
    jwks_raw = credentials.get("jwt_jwks")
    if not jwks_raw:
        logger.error("jwt_jwks is missing from credentials")
        raise ValueError("jwt_jwks is required when auth_method is jwt")
    if isinstance(jwks_raw, dict):
        logger.debug("JWKS is already a dict")
        return jwks_raw
    try:
        jwks = json.loads(jwks_raw)
        logger.debug(f"Successfully parsed JWKS, kty: {jwks.get('kty')}")
        return jwks
    except Exception as ex:  # pragma: no cover - defensive
        logger.error(f"Failed to parse JWKS JSON: {ex}")
        raise ValueError(f"Invalid JWKS JSON: {ex}") from ex


def extract_jwt_signing_key(jwks: Dict[str, Any]) -> Any:
    """Extract JWT signing key from JWKS."""
    kty = jwks.get("kty")
    logger.debug(f"Extracting JWT signing key from JWKS, kty: {kty}")

    # Symmetric key (oct)
    if jwks.get("kty") == "oct" and "k" in jwks:
        logger.debug("Using symmetric key (oct) from JWKS")
        # JWKS k field is base64url encoded
        padded = jwks["k"] + "=="  # ensure correct padding
        key = base64.urlsafe_b64decode(padded)
        logger.debug(f"Decoded symmetric key, length: {len(key)} bytes")
        return key

    # Allow direct private key string in JWKS under `private_key`
    if "private_key" in jwks:
        logger.debug("Using private_key from JWKS")
        return jwks["private_key"]

    available_keys = list(jwks.keys())
    logger.error(f"Unsupported JWKS format, available keys: {available_keys}")
    raise ValueError("Unsupported JWKS format: missing signing key")


def generate_jwt_token(credentials: dict) -> str:
    """Generate JWT token from credentials."""
    logger.info("Generating JWT token")

    alg = credentials.get("jwt_algorithm", "HS256")
    consumer_id = credentials.get("jwt_consumer_id")
    logger.debug(f"JWT algorithm: {alg}, consumer_id: {consumer_id}")

    if not consumer_id:
        logger.error("jwt_consumer_id is missing")
        raise ValueError("jwt_consumer_id (uid) is required for JWT auth")

    expiration_seconds = int(credentials.get("jwt_expiration", 7200))
    expiration_seconds = min(expiration_seconds, 604800)  # max 7 days
    logger.debug(f"JWT expiration: {expiration_seconds} seconds")

    now = int(time.time())
    payload: Dict[str, Any] = {
        "jti": str(now),
        "iat": now,
        "nbf": now - 60,
        "exp": now + expiration_seconds,
        "uid": consumer_id,
    }
    issuer = credentials.get("jwt_issuer")
    if issuer:
        payload["iss"] = issuer
        logger.debug(f"JWT issuer: {issuer}")

    jti = payload['jti']
    iat = payload['iat']
    exp = payload['exp']
    logger.debug(f"JWT payload: jti={jti}, iat={iat}, exp={exp}")

    jwks = parse_jwks(credentials)
    signing_key = extract_jwt_signing_key(jwks)

    token = jwt.encode(payload, signing_key, algorithm=alg)
    token_len = len(token)
    logger.info(f"JWT token generated successfully, length: {token_len}")
    logger.debug(f"JWT token (first 20 chars): {token[:20]}...")
    return token


def build_hmac_headers(
    method: str,
    path: str,
    body: bytes,
    access_key: str,
    secret_key: str,
    signature_headers: Optional[str] = None,
) -> Dict[str, str]:
    """Build HMAC authentication headers."""
    logger.info(f"Building HMAC headers for {method} {path}")
    body_len = len(body)
    logger.debug(f"Access key: {access_key}, body length: {body_len} bytes")

    accept = "application/json"
    content_type = "application/json"
    date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    logger.debug(f"Date: {date_str}")

    md5_hasher = hashlib.md5()
    md5_hasher.update(body)
    content_md5 = base64.b64encode(md5_hasher.digest()).decode()
    logger.debug(f"Content-MD5: {content_md5}")

    canonical_headers = ""
    signature_headers_value = ""
    if signature_headers:
        names = [h.strip() for h in signature_headers.split(",") if h.strip()]
        names.sort()
        canonical_headers = "\n".join([f"{n}:{''}" for n in names])
        signature_headers_value = ",".join(names)
        logger.debug(f"Signature headers: {signature_headers_value}")

    to_sign_parts = [
        method,
        accept,
        content_md5,
        content_type,
        date_str,
    ]
    if canonical_headers:
        to_sign_parts.append(canonical_headers)
    else:
        to_sign_parts.append("")
    to_sign_parts.append(path)
    string_to_sign = "\n".join(to_sign_parts)

    logger.debug(f"String to sign:\n{string_to_sign}")

    signer = hmac.new(
        secret_key.encode(),
        string_to_sign.encode(),
        hashlib.sha256,
    )
    signature = base64.b64encode(signer.digest()).decode()
    logger.debug(f"HMAC signature: {signature}")

    headers = {
        "accept": accept,
        "content-type": content_type,
        "date": date_str,
        "content-md5": content_md5,
        "x-ca-key": access_key,
        "x-ca-signature": signature,
        "x-ca-signature-headers": signature_headers_value,
    }

    sig_len = len(signature)
    logger.info(f"HMAC headers built successfully, signature len: {sig_len}")
    return headers


def prepare_auth_headers(
    credentials: dict,
    method: str,
    path: str,
    body: bytes,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Prepare authentication headers based on auth_method.

    :param credentials: Model credentials
    :param method: HTTP method (e.g., "POST")
    :param path: API path (e.g., "/chat/completions")
    :param body: Request body bytes
    :param extra_headers: Existing extra headers dict (will be modified)
    :return: Updated headers dict
    """
    if extra_headers is None:
        extra_headers = {}

    auth_method = credentials.get("auth_method")
    logger.info(f"Preparing auth headers using method: {auth_method}")
    body_len = len(body)
    logger.debug(f"Request: {method} {path}, body length: {body_len} bytes")

    if not auth_method:
        logger.error("auth_method is missing from credentials")
        raise ValueError("auth_method is required (api_key, jwt, or hmac)")

    if auth_method == "jwt":
        logger.info("Using JWT authentication")
        token = generate_jwt_token(credentials)
        header_name = credentials.get("jwt_header_name") or "Authorization"
        header_prefix = credentials.get("jwt_header_prefix") or "Bearer"
        header_value = f"{header_prefix} {token}".strip()
        extra_headers[header_name] = header_value
        logger.debug(f"JWT header: {header_name}: {header_prefix} [token]")
        # Preserve compatibility with base class Authorization handling
        credentials["api_key"] = token

    elif auth_method == "hmac":
        logger.info("Using HMAC authentication")
        access_key = credentials.get("hmac_access_key")
        secret_key = credentials.get("hmac_secret_key")
        if not access_key or not secret_key:
            logger.error("HMAC credentials are incomplete")
            err_msg = (
                "hmac_access_key and hmac_secret_key are required "
                "for HMAC auth"
            )
            raise ValueError(err_msg)

        hmac_headers = build_hmac_headers(
            method=method,
            path=path,
            body=body,
            access_key=access_key,
            secret_key=secret_key,
            signature_headers=None,
        )
        extra_headers.update(hmac_headers)
        logger.debug(f"HMAC headers added: {list(hmac_headers.keys())}")

        # Ensure Authorization from base is not used
        credentials["api_key"] = ""

    elif auth_method == "api_key":
        logger.info("Using API Key authentication")
        api_key = credentials.get("api_key", "")
        if api_key:
            logger.debug(f"API key present, length: {len(api_key)} chars")
        else:
            logger.warning("API key is empty")
        # base class will handle Authorization header using api_key
        pass
    else:  # pragma: no cover - defensive
        logger.error(f"Unsupported auth_method: {auth_method}")
        raise ValueError(f"Unsupported auth_method: {auth_method}")

    header_count = len(extra_headers)
    logger.info(
        f"Auth headers prepared successfully, total headers: {header_count}"
    )
    return extra_headers


def get_api_path(credentials: dict, default_path: str) -> str:
    """
    Get API path from credentials, handling endpoint_url and mode.

    :param credentials: Model credentials
    :param default_path: Default path if endpoint_url is not set
    :return: Full API path
    """
    endpoint_url = credentials.get("endpoint_url", "")
    logger.debug(
        f"Getting API path, endpoint_url: {endpoint_url}, "
        f"default_path: {default_path}"
    )

    if not endpoint_url:
        logger.debug(f"No endpoint_url, using default path: {default_path}")
        return default_path

    parsed = urlparse(endpoint_url)
    base_path = parsed.path.rstrip("/")
    path = (base_path or "") + default_path
    if not path.startswith("/"):
        path = "/" + path

    logger.debug(f"Resolved API path: {path}")
    return path
