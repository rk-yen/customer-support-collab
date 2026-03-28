import os

from arize.otel import register
from google.colab import userdata
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

WORKSHOP_PROJECT_NAME = "Workiva-ProblemFirst-Workshop"


def setup_clients(project_name: str = WORKSHOP_PROJECT_NAME) -> dict:
    """Initialize Arize tracing and the OpenAI client inside Colab."""
    arize_space_id = userdata.get("ARIZE_SPACE_ID")
    arize_api_key = userdata.get("ARIZE_API_KEY")
    openai_api_key = userdata.get("OPENAI_API_KEY")

    tracer_provider = register(
        space_id=arize_space_id,
        api_key=arize_api_key,
        project_name=project_name,
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    os.environ["OPENAI_API_KEY"] = openai_api_key

    return {
        "client": OpenAI(),
        "arize_space_id": arize_space_id,
        "arize_api_key": arize_api_key,
        "tracer_provider": tracer_provider,
    }

