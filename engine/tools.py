import logging
from typing import Annotated
from livekit.agents import llm

logger = logging.getLogger("agent-tools")

class AssistantFnc(llm.FunctionContext):
    def __init__(self):
        super().__init__()

    @llm.ai_callable(description="Get the current weather for a specific location")
    def get_weather(self, location: Annotated[str, llm.TypeInfo(description="The city and state, e.g. San Francisco, CA")]):
        logger.info(f"[AGENT_DECISION] tool_selected: get_weather, args: {location}")
        if not location:
            logger.error("[AGENT_ERROR] tool_failed: get_weather, reason: empty location")
            return "Error: location is required"
        logger.info(f"[AGENT_ACTION] tool_started: get_weather for {location}")
        # Mock implementation for safety
        result = f"The weather in {location} is currently 72 degrees and sunny."
        logger.info(f"[AGENT_RESULT] tool_completed: get_weather, result: {result}")
        return result

