import os
from contextlib import contextmanager
from datetime import datetime

from arize.otel import register
RUNNING_ON_GOOGLE_COLAB = True
try:
    from google.colab import userdata    
except Exception:
    from dotenv import load_dotenv
    RUNNING_ON_GOOGLE_COLAB = False
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

WORKSHOP_PROJECT_NAME = f"Workiva-ProblemFirst-Workshop-{datetime.now().strftime('%Y-%m-%d')}"
_TRACE_RUNTIME: dict = {
    "tracer_provider": None,
    "openai_instrumentor": None,
    "agents_instrumentor": None,
}


def _clear_existing_instrumentation() -> None:
    for key in ("openai_instrumentor", "agents_instrumentor"):
        instrumentor = _TRACE_RUNTIME.get(key)
        if instrumentor is None:
            continue
        try:
            instrumentor.uninstrument()
        except Exception:
            pass


def setup_clients(project_name: str = WORKSHOP_PROJECT_NAME) -> dict:
    """Initialize Arize tracing and the OpenAI client."""
    if RUNNING_ON_GOOGLE_COLAB:
        arize_space_id = userdata.get("ARIZE_SPACE_ID")
        arize_api_key = userdata.get("ARIZE_API_KEY")
        openai_api_key = userdata.get("OPENAI_API_KEY")
    else:
        load_dotenv()
        arize_space_id = os.getenv("ARIZE_SPACE_ID")
        arize_api_key = os.getenv("ARIZE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
    tracer_provider = register(
        space_id=arize_space_id,
        api_key=arize_api_key,
        project_name=project_name,
        set_global_tracer_provider=False,
    )
    os.environ["OPENAI_API_KEY"] = openai_api_key

    _clear_existing_instrumentation()
    openai_instrumentor = OpenAIInstrumentor()
    agents_instrumentor = OpenAIAgentsInstrumentor()
    openai_instrumentor.instrument(tracer_provider=tracer_provider)
    agents_instrumentor.instrument(tracer_provider=tracer_provider)

    _TRACE_RUNTIME.update(
        {
            "tracer_provider": tracer_provider,
            "openai_instrumentor": openai_instrumentor,
            "agents_instrumentor": agents_instrumentor,
        }
    )

    return {
        "client": OpenAI(),
        "arize_space_id": arize_space_id,
        "arize_api_key": arize_api_key,
        "tracer_provider": tracer_provider,
    }


@contextmanager
def suspend_openai_tracing_for_agents():
    """Temporarily disable the generic OpenAI instrumentor around Agents SDK runs."""
    openai_instrumentor = _TRACE_RUNTIME.get("openai_instrumentor")
    tracer_provider = _TRACE_RUNTIME.get("tracer_provider")

    if openai_instrumentor is None or tracer_provider is None:
        yield
        return

    openai_instrumentor.uninstrument()
    try:
        yield
    finally:
        openai_instrumentor.instrument(tracer_provider=tracer_provider)
