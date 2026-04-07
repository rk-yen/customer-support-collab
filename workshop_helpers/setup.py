import os
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

WORKSHOP_PROJECT_NAME = f"Workiva-ProblemFirst-Workshop-{datetime.now().strftime('%Y-%m-%d')}"


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
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    os.environ["OPENAI_API_KEY"] = openai_api_key

    return {
        "client": OpenAI(),
        "arize_space_id": arize_space_id,
        "arize_api_key": arize_api_key,
        "tracer_provider": tracer_provider,
    }

