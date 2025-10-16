from agent_config.context import CustomerSupportContext
from agent_config.access_token import get_gateway_access_token
from agent_config.agent_task import agent_task
from agent_config.streaming_queue import StreamingQueue
from agent_config.ssm import get_ssm_parameter 
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import asyncio
import logging
import os


# Environment flags
os.environ["STRANDS_OTEL_ENABLE_CONSOLE_EXPORT"] = "true"
os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"

os.environ["KNOWLEDGE_BASE_ID"] = 'I7A05RY78W'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bedrock app and global agent instance
app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload, context):
    if not CustomerSupportContext.get_response_queue_ctx():
        CustomerSupportContext.set_response_queue_ctx(StreamingQueue())

    if not CustomerSupportContext.get_gateway_token_ctx():
        CustomerSupportContext.set_gateway_token_ctx(
           await get_gateway_access_token()
        )
    user_message = payload["prompt"]
    actor_id = payload["actor_id"]

    session_id = context.session_id

    if not session_id:
        raise Exception("Context session_id is not set")
    print(f"actor_id={actor_id}")
    task = asyncio.create_task(
        agent_task(
            user_message=user_message,
            session_id=session_id,
            actor_id=actor_id,
        )
    )

    response_queue = CustomerSupportContext.get_response_queue_ctx()

    async def stream_output():
        async for item in response_queue.stream():
            yield item
        await task  # Ensure task completion

    return stream_output()


if __name__ == "__main__":
    app.run()