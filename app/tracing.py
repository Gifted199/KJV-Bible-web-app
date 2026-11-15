"""Langfuse tracing: one trace per question, with a child span for each step.

    kjv-answer (trace)
    ├── input-guard            placeholder until phase 5: always passes
    ├── question-embedding     text-embedding-3-small, or the local cache (an embedding call only on a cache miss)
    ├── ai-search              AI Search hybrid: the candidates and their fused scores
    ├── bge-rerank             the candidates re-scored by bge, best first
    ├── abstention-check       best bge score against the threshold
    ├── generation             gpt-5.4-mini (the Langfuse OpenAI integration records tokens and cost); twice on a retry
    └── citation-verification  the quote checks
    scores: abstained, top_bge_score, passages, citation_checks_passed, llm_calls, cost_usd

Switch: tracing is on when the Langfuse keys are set, and off when LANGFUSE_TRACING_ENABLED=false (the tests set it).
When off, every function here does nothing, so the pipeline code calls them unconditionally.
Keys are never sent: every string is passed through config.redact before it leaves the machine.
"""

import os
from contextlib import contextmanager

from app import config

OFF_VALUES = {"false", "0", "no", "off"}


def enabled():
    config.load_env_file()
    switched_off = os.environ.get("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in OFF_VALUES
    keys = all((os.environ.get(k) or "").strip() for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"))
    return keys and not switched_off


def mask(*, data, **kwargs):
    """Langfuse's mask hook: replace any secret value inside the data before it is sent."""
    if isinstance(data, str):
        return config.redact(data)
    if isinstance(data, dict):
        return {k: mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(data=v) for v in data]
    return data


class NoSpan:
    """What the span functions yield when tracing is off."""

    trace_id = None
    id = None

    def update(self, **fields):
        pass

    def score(self, **fields):
        pass

    def score_trace(self, **fields):
        pass


@contextmanager
def trace(name, input=None, metadata=None, tags=None):
    """The root observation of one trace; yields it (its trace_id is the id to look up in Langfuse)."""
    if not enabled():
        yield NoSpan()
        return
    from langfuse import propagate_attributes

    with config.langfuse_client().start_as_current_observation(name=name, as_type="span", input=input,
                                                               metadata=metadata) as root:
        with propagate_attributes(trace_name=name, tags=tags):
            yield root


@contextmanager
def span(name, as_type="span", input=None, metadata=None):
    """A child of the current trace. as_type: span, retriever, guardrail, embedding, generation, ...

    Outside a trace (e.g. an evaluation that calls the retriever directly) it records nothing, so batch runs never
    scatter loose spans across Langfuse.
    """
    if not enabled() or config.langfuse_client().get_current_trace_id() is None:
        yield NoSpan()
        return
    with config.langfuse_client().start_as_current_observation(name=name, as_type=as_type, input=input,
                                                               metadata=metadata) as current:
        yield current


def disable():
    """Switch tracing off for this process (before any client is created): used by the commands that don't trace."""
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"


def llm_kwargs(name, metadata=None):
    """Extra arguments for an OpenAI call: the Langfuse integration reads (and removes) them; nothing when off."""
    if not enabled():
        return {}
    return {"name": name, **({"metadata": metadata} if metadata else {})}


def flush():
    """Send buffered traces; the CLI calls this before it exits."""
    if enabled():
        config.langfuse_client().flush()
