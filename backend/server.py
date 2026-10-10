from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("digital_twin")

app = FastAPI(title="AI Digital Twin API", version="1.0.0")

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# AI Provider configuration
#
# Set  AI_PROVIDER=bedrock  (default) or  AI_PROVIDER=openai  in .env.
# When AI_PROVIDER is not set the server tries Bedrock first and
# automatically falls back to OpenAI if access is denied or the model
# is unavailable, so no manual intervention is normally required.
# ---------------------------------------------------------------------------

AI_PROVIDER = os.getenv("AI_PROVIDER", "auto").lower()   # "bedrock" | "openai" | "auto"

# --- Bedrock ---
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.amazon.nova-2-lite-v1:0")
bedrock_client: Optional[boto3.client] = None

try:
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("DEFAULT_AWS_REGION", "us-east-1"),
    )
except Exception as exc:
    logger.warning("Could not initialise Bedrock client: %s", exc)

# --- OpenAI ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_client = None

try:
    from openai import OpenAI
    _openai_key = os.getenv("OPENAI_API_KEY", "")
    if _openai_key:
        openai_client = OpenAI(api_key=_openai_key)
        logger.info("OpenAI client initialised (model: %s)", OPENAI_MODEL)
    else:
        logger.warning("OPENAI_API_KEY is not set � OpenAI fallback unavailable.")
except ImportError:
    logger.warning("openai package not installed � OpenAI fallback unavailable.")

# ---------------------------------------------------------------------------
# Memory / storage configuration
# ---------------------------------------------------------------------------

USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

s3_client = None
if USE_S3:
    s3_client = boto3.client("s3")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    provider: str   # tells the caller which backend was used


class Message(BaseModel):
    role: str
    content: str
    timestamp: str

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from S3 or local disk."""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]) -> None:
    """Persist conversation history to S3 or local disk."""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2)

# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_bedrock(conversation: List[Dict], user_message: str) -> str:
    """Send a message to AWS Bedrock and return the text reply."""
    if bedrock_client is None:
        raise RuntimeError("Bedrock client is not available.")

    # Build the alternating user/assistant message list Bedrock expects.
    messages = [
        {"role": msg["role"], "content": [{"text": msg["content"]}]}
        for msg in conversation[-50:]
    ]
    messages.append({"role": "user", "content": [{"text": user_message}]})

    response = bedrock_client.converse(
        modelId=BEDROCK_MODEL_ID,
        system=[{"text": prompt()}],
        messages=messages,
        inferenceConfig={"maxTokens": 2000, "temperature": 0.7, "topP": 0.9},
    )
    return response["output"]["message"]["content"][0]["text"]


def _call_openai(conversation: List[Dict], user_message: str) -> str:
    """Send a message to OpenAI Chat Completions and return the text reply."""
    if openai_client is None:
        raise RuntimeError("OpenAI client is not available. Set OPENAI_API_KEY in .env.")

    # Build the standard OpenAI messages list.
    messages: List[Dict] = [{"role": "system", "content": prompt()}]
    for msg in conversation[-50:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.9,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Provider router � the single function the endpoint calls
# ---------------------------------------------------------------------------

_BEDROCK_FALLBACK_ERRORS = {
    "AccessDeniedException",
    "ResourceNotFoundException",
    "ServiceUnavailableException",
    "ThrottlingException",
}


def call_ai(conversation: List[Dict], user_message: str) -> tuple[str, str]:
    """
    Route the request to the configured AI provider.

    Returns:
        (reply_text, provider_name)  � provider_name is "bedrock" or "openai".

    Strategy:
        AI_PROVIDER=bedrock  ? Bedrock only, raises on error.
        AI_PROVIDER=openai   ? OpenAI only, raises on error.
        AI_PROVIDER=auto     ? Try Bedrock; fall back to OpenAI on access /
                               availability errors so the service keeps
                               running without any code changes.
    """
    if AI_PROVIDER == "openai":
        logger.info("Using OpenAI (forced by AI_PROVIDER env var).")
        return _call_openai(conversation, user_message), "openai"

    if AI_PROVIDER == "bedrock":
        logger.info("Using Bedrock (forced by AI_PROVIDER env var).")
        return _call_bedrock(conversation, user_message), "bedrock"

    # ---- auto mode --------------------------------------------------------
    try:
        logger.info("Auto mode: attempting Bedrock (%s).", BEDROCK_MODEL_ID)
        reply = _call_bedrock(conversation, user_message)
        logger.info("Bedrock responded successfully.")
        return reply, "bedrock"

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in _BEDROCK_FALLBACK_ERRORS:
            logger.warning(
                "Bedrock unavailable (%s: %s). Falling back to OpenAI.",
                error_code,
                exc.response["Error"]["Message"],
            )
        else:
            # Unexpected Bedrock error � do not silently swallow it.
            logger.error("Unexpected Bedrock error (%s): %s", error_code, exc)
            raise HTTPException(status_code=500, detail=f"Bedrock error: {exc}")

    except RuntimeError as exc:
        # Bedrock client was never initialised.
        logger.warning("Bedrock client unavailable (%s). Falling back to OpenAI.", exc)

    # Fallback
    logger.info("Using OpenAI fallback (model: %s).", OPENAI_MODEL)
    return _call_openai(conversation, user_message), "openai"

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
        "ai_provider": AI_PROVIDER,
        "bedrock_model": BEDROCK_MODEL_ID,
        "openai_model": OPENAI_MODEL,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "use_s3": USE_S3,
        "ai_provider": AI_PROVIDER,
        "bedrock_model": BEDROCK_MODEL_ID,
        "openai_model": OPENAI_MODEL,
        "openai_available": openai_client is not None,
        "bedrock_available": bedrock_client is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        conversation = load_conversation(session_id)

        assistant_response, provider_used = call_ai(conversation, request.message)

        now = datetime.now().isoformat()
        conversation.append({"role": "user", "content": request.message, "timestamp": now})
        conversation.append({"role": "assistant", "content": assistant_response, "timestamp": now})

        save_conversation(session_id, conversation)

        logger.info("Session %s � replied via %s.", session_id, provider_used)
        return ChatResponse(
            response=assistant_response,
            session_id=session_id,
            provider=provider_used,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /chat: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve full conversation history for a session."""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
