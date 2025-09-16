import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from my_tools import get_tool_placeholder_message
from query_agent import get_agent_executor, get_db


# Reuse the agent executor across warm invocations
_agent_executor = None


def _get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = get_agent_executor()
    return _agent_executor


def _extract_params(event: Dict[str, Any]) -> Dict[str, Optional[str]]:
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod"))

    question: Optional[str] = None
    thread_id: Optional[str] = None
    cp_access_token: Optional[str] = None

    # Prefer query string params for GET
    if event.get("queryStringParameters"):
        question = event["queryStringParameters"].get("question")
        thread_id = event["queryStringParameters"].get("thread_id")

    # Fallback to JSON body (POST)
    if (question is None or thread_id is None) and event.get("body"):
        try:
            body = event["body"]
            if event.get("isBase64Encoded"):
                # In API Gateway, body may be base64-encoded; Lambda decodes automatically only when using certain integrations
                import base64
                body = base64.b64decode(body).decode("utf-8")
            payload = json.loads(body)
            question = question or payload.get("question")
            thread_id = thread_id or payload.get("thread_id")
        except Exception:
            # Ignore body parsing errors; validation below will catch missing fields
            pass

    # Extract cookies
    cp_access_token = _get_cookie_value(event, "cp_access_token")

    return {
        "method": method,
        "question": question,
        "thread_id": thread_id,
        "cp_access_token": cp_access_token,
    }


def _get_cookie_value(event: Dict[str, Any], name: str) -> Optional[str]:
    """Extract a cookie value from an API Gateway event.

    Supports both API Gateway v2 (HTTP API) via event["cookies"] and
    REST API via event["headers"]["Cookie"].
    """
    # API Gateway v2: event["cookies"] is a list of cookie header strings
    cookies_list = event.get("cookies")
    if isinstance(cookies_list, list) and cookies_list:
        # Join multiple cookie header lines and parse
        combined = "; ".join([c for c in cookies_list if isinstance(c, str)])
        value = _parse_cookie_header_for_name(combined, name)
        if value is not None:
            return value

    # API Gateway REST or generic: headers["cookie"] or headers["Cookie"]
    headers = event.get("headers") or {}
    if isinstance(headers, dict):
        cookie_header = headers.get("cookie") or headers.get("Cookie")
        if isinstance(cookie_header, str):
            return _parse_cookie_header_for_name(cookie_header, name)

    return None


def _parse_cookie_header_for_name(cookie_header: str, name: str) -> Optional[str]:
    if not cookie_header:
        return None
    parts = cookie_header.split(";")
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if "=" not in piece:
            continue
        key, val = piece.split("=", 1)
        if key.strip() == name:
            return val
    return None


def _event_to_line(message: AIMessage) -> str:
    # Mirror logic from Flask app: emit {"message": content or placeholder(tool_name)}
    content = message.content
    if not content:
        tool_calls = (message.additional_kwargs or {}).get("tool_calls", [])
        tool_name = None
        if tool_calls and isinstance(tool_calls, list):
            call0 = tool_calls[0] or {}
            func = call0.get("function", {}) if isinstance(call0, dict) else {}
            tool_name = func.get("name")
        content = get_tool_placeholder_message(tool_name)
    return json.dumps({"message": content})

def _validate_access_token(access_token: Optional[str]) -> Dict[str, Any]:
    """Validate and decode the provided access token.

    Replicates the Ruby flow:
      1) validate_access_token(access_token) -> {} or {errors: [...]}
      2) decoded_access_token = Decoder.decode(access_token)
         - if errors include "Signature has expired" then revoke(token)
         - return decoded or errors
      3) authenticate_tokenable_from_token(decoded)

    Returns a dict. If there's an error, it will include key "errors" with a list.
    On success, returns an empty dict {} (or the result of authenticate step if it returns errors).
    """
    if not access_token:
        return _jwt_error(["Missing access token"])

    # Step 1: validate existence/allowlist of token (placeholder hook)
    if not _is_access_token_known(access_token):
        return _invalid_token_error()

    # Step 2: decode with RS256 using configured public key
    decoded_or_error = _decode_jwt(access_token)
    if isinstance(decoded_or_error, dict) and decoded_or_error.get("errors"):
        errors = decoded_or_error["errors"]
        if any("Signature has expired" in e for e in errors):
            _revoke_token(access_token)
        return decoded_or_error

    decoded_claims = decoded_or_error if isinstance(decoded_or_error, dict) else {}

    # Step 3: authenticate subject/principal from token (placeholder hook)
    auth_result = _authenticate_tokenable_from_token(decoded_claims)
    if auth_result.get("errors"):
        return auth_result

    return {}


def _invalid_token_error() -> Dict[str, Any]:
    return _jwt_error(["Invalid token"])


def _jwt_error(messages: List[str]) -> Dict[str, Any]:
    return {"errors": messages}


def _get_jwt_public_key() -> Optional[str]:
    """Fetch RSA public key for verifying JWT (PEM string)."""
    return os.environ.get("JWT_PUBLIC_KEY")


def _decode_jwt(token: str) -> Dict[str, Any]:
    """Decode JWT with RS256 using public key.

    Returns payload dict on success, or {errors: [...]} on failure.
    """
    try:
        import jwt  # PyJWT
    except Exception:
        return _jwt_error(["JWT library not available (pyjwt)"])

    public_key = _get_jwt_public_key()
    if not public_key:
        return _jwt_error(["JWT public key not configured (JWT_PUBLIC_KEY)"])

    try:
        payload = jwt.decode(token, public_key, algorithms=["RS256"])
        if isinstance(payload, dict):
            return payload
        return {}
    except Exception as error:  # Mirror broad rescue in Ruby
        message = str(error)
        # Normalize common expiration message
        if "expired" in message.lower():
            message = "Signature has expired"
        return _jwt_error([message])


def _revoke_token(token: str) -> None:
    """Hook to revoke token (placeholder).

    In Ruby: JwtAuthentication::Revoker.revoke(access_token)
    Implement actual revocation if needed (e.g., write to blocklist).
    """
    return None


def _is_access_token_known(token: str) -> bool:
    """Check if access token exists in DB (mimics AccessToken.valid.find_by(...).present?).

    This uses the same PostgreSQL connection configuration as `query_agent.get_db()`.
    The minimal implementation checks presence by `encoded_token` in `access_tokens`.
    Enhance with validity filters to match your Rails `valid` scope if needed.
    """
    if not token:
        return False

    try:
        db = get_db()
        engine = getattr(db, "engine", None) or getattr(db, "_engine", None)
        if engine is None:
            return False
        result_str = db.run("SELECT 1 FROM access_tokens WHERE is_valid = True AND encoded_token = '" + token.replace("'", "''") + "' LIMIT 1;")
        return bool(result_str and "1" in result_str)

    except Exception:
        return False


def _authenticate_tokenable_from_token(decoded_claims: Dict[str, Any]) -> Dict[str, Any]:
    """Authenticate principal from decoded claims (placeholder).

    In Ruby: authenticate_tokenable_from_token(decoded_access_token)
    Return {} on success or {errors: [...]} on failure.
    """
    # Example check: ensure required subject or uid exists
    subject = decoded_claims.get("sub") if isinstance(decoded_claims, dict) else None
    if not subject:
        return _jwt_error(["Token subject missing"])
    return {}

def lambda_handler(event: Dict[str, Any], context: Any, response_stream: Any = None) -> Dict[str, Any]:
    params = _extract_params(event)
    question = params.get("question")
    thread_id = params.get("thread_id")

    if not question:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Missing 'question' parameter"}),
        }

    # thread_id is optional but supported
    agent_executor = _get_agent_executor()

    try:
        events = agent_executor.stream(
            {"messages": [("user", question)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        )

        # If Lambda Response Streaming is enabled and a response_stream is provided,
        # write chunks incrementally as NDJSON lines.
        if response_stream is not None and hasattr(response_stream, "write"):
            # CORS + streaming friendly headers
            headers = {
                "Content-Type": "application/json",  # NDJSON
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            }

            for event_item in events:
                last_message = event_item["messages"][-1]
                if isinstance(last_message, AIMessage):
                    line = _event_to_line(last_message) + "\n"
                    # Write bytes as per streaming API
                    response_stream.write(line.encode("utf-8"))
            # Return headers and status; body is omitted when streaming
            return {"statusCode": 200, "headers": headers}

        # Fallback: Accumulate and return in one response
        lines: List[str] = []
        for event_item in events:
            last_message = event_item["messages"][-1]
            if isinstance(last_message, AIMessage):
                lines.append(_event_to_line(last_message))

        body = "\n".join(lines) + ("\n" if lines else "")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            },
            "body": body,
            "isBase64Encoded": False,
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Internal Server Error", "details": str(e)}),
        }


