"""Settings and clients for the Azure and Langfuse services.

Values come from environment variables: from `.env` when running locally (git-ignored), and from Key Vault-backed
environment variables on Azure. Secrets are never printed: errors name the missing variable, never a value, and any
error text from a service has the secret values masked before it is shown.

Each service has its own group of settings, so a command only needs the services it uses:

    openai    one Azure OpenAI resource with two deployments:
              gpt-5.4-mini (writes the answers and judges them) and text-embedding-3-small (1,536 dimensions)
    search    Azure AI Search (free tier), index of the 4,723 parents
    langfuse  Langfuse Cloud (EU), tracing and evaluation

The client uses Azure's v1 API (`<endpoint>/openai/v1/`), which needs no api-version. gpt-5.4-mini only answered there.
"""

import os
from functools import cache
from urllib.parse import urlparse

EMBED_DIMENSIONS = 1536  # text-embedding-3-small at full size
LANGFUSE_EU = "https://cloud.langfuse.com"
#seconds; without them a request that loses its connection (e.g. the laptop sleeps) can wait for hours, not retry
SEARCH_TIMEOUTS = {"connection_timeout": 10, "read_timeout": 30}

GROUPS = {
    "openai": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT",
               "AZURE_OPENAI_EMBED_DEPLOYMENT"],
    "search": ["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX"],
    "langfuse": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
}
SECRETS = ["AZURE_OPENAI_API_KEY", "AZURE_SEARCH_API_KEY", "LANGFUSE_SECRET_KEY"]

#what gpt-5.4-mini accepts (tested on the deployment): max_completion_tokens, never max_tokens; temperature and top_p only
#while reasoning_effort is "none" (the default); with any reasoning on, only the default temperature of 1 is allowed
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


class MissingSettings(RuntimeError):
    pass


@cache
def load_env_file():
    """Read `.env` once. Variables already set in the environment win, so Azure's settings are never overridden."""
    from dotenv import find_dotenv, load_dotenv

    #search from the working folder up, so it also works in notebooks and piped scripts
    load_dotenv(find_dotenv(usecwd=True), override=False)


def require(group, env=None):
    """The settings for one service, as a dict; raises MissingSettings naming every empty variable."""
    if env is None:
        load_env_file()
        env = os.environ
    values = {name: (env.get(name) or "").strip() for name in GROUPS[group]}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise MissingSettings(f"{group}: set {', '.join(missing)} in .env (see .env.example)")
    return values


def redact(text, env=None):
    """Mask every secret value that appears in the text, e.g. in an error message from a service."""
    env = os.environ if env is None else env
    for name in SECRETS:
        value = (env.get(name) or "").strip()
        if len(value) >= 8:
            text = text.replace(value, "***")
    return text


def base_endpoint(url):
    """https://host from any URL the portal shows, including a full "Target URI" with a path and api-version."""
    parts = urlparse(url.strip())
    if parts.scheme != "https" or not parts.hostname:
        raise MissingSettings("AZURE_OPENAI_ENDPOINT must look like https://<resource>.openai.azure.com/")
    return f"https://{parts.hostname}"


def chat_options(max_completion_tokens, reasoning_effort="none", temperature=None):
    """Request settings for gpt-5.4-mini, refusing the combinations the model rejects."""
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
    if temperature is not None and reasoning_effort != "none" and temperature != 1:
        raise ValueError("gpt-5.4-mini accepts a temperature other than 1 only when reasoning_effort is 'none'")
    options = {"max_completion_tokens": max_completion_tokens, "reasoning_effort": reasoning_effort}
    if temperature is not None:
        options["temperature"] = temperature
    return options


def chat_deployment():
    return require("openai")["AZURE_OPENAI_CHAT_DEPLOYMENT"]


def embed_deployment():
    return require("openai")["AZURE_OPENAI_EMBED_DEPLOYMENT"]


@cache
def openai_client():
    """One client for both deployments: they share the resource, endpoint and key.

    With tracing on it is Langfuse's drop-in OpenAI client, which records every call's tokens and cost.
    """
    from app import tracing

    if tracing.enabled():
        from langfuse.openai import OpenAI
    else:
        from openai import OpenAI

    s = require("openai")
    return OpenAI(base_url=f"{base_endpoint(s['AZURE_OPENAI_ENDPOINT'])}/openai/v1/", api_key=s["AZURE_OPENAI_API_KEY"])


def search_index_name():
    return require("search")["AZURE_SEARCH_INDEX"]


@cache
def search_index_client():
    """Creates and deletes indexes."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient

    s = require("search")
    return SearchIndexClient(s["AZURE_SEARCH_ENDPOINT"], AzureKeyCredential(s["AZURE_SEARCH_API_KEY"]), **SEARCH_TIMEOUTS)


@cache
def search_client():
    """Uploads documents to the index and searches it."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    s = require("search")
    return SearchClient(s["AZURE_SEARCH_ENDPOINT"], s["AZURE_SEARCH_INDEX"], AzureKeyCredential(s["AZURE_SEARCH_API_KEY"]),
                        **SEARCH_TIMEOUTS)


def langfuse_base_url():
    load_env_file()
    return (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or LANGFUSE_EU).strip()


@cache
def langfuse_client():
    from langfuse import Langfuse

    s = require("langfuse")
    from app.tracing import mask

    return Langfuse(public_key=s["LANGFUSE_PUBLIC_KEY"], secret_key=s["LANGFUSE_SECRET_KEY"],
                    base_url=langfuse_base_url(), mask=mask)


def check_chat():
    response = openai_client().chat.completions.create(
        model=chat_deployment(), messages=[{"role": "user", "content": "Reply with the single word OK."}],
        **chat_options(max_completion_tokens=20),
    )
    return f"deployment {chat_deployment()} answered (model {response.model})"


def check_embeddings():
    vector = openai_client().embeddings.create(model=embed_deployment(), input=["In the beginning"]).data[0].embedding
    if len(vector) != EMBED_DIMENSIONS:
        raise RuntimeError(f"expected {EMBED_DIMENSIONS} dimensions, got {len(vector)}")
    return f"deployment {embed_deployment()} returned {len(vector)} dimensions"


def check_search():
    names = list(search_index_client().list_index_names())
    status = "exists" if search_index_name() in names else "not created yet (phase 1 creates it)"
    return f"service reachable, {len(names)} index(es); '{search_index_name()}' {status}"


def check_langfuse():
    if not langfuse_client().auth_check():
        raise RuntimeError("keys rejected")
    return f"keys accepted by {langfuse_base_url()}"


CHECKS = {"chat": check_chat, "embeddings": check_embeddings, "search": check_search, "langfuse": check_langfuse}


def check_connections():
    """Try each service once; yields (service, ok, message). Messages never contain a secret."""
    for name, check in CHECKS.items():
        try:
            yield name, True, check()
        except Exception as error:  # report every service, even after one fails
            message = redact(f"{type(error).__name__}: {error}")
            yield name, False, message[:300]
