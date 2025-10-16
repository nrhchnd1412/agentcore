from .utils import get_ssm_parameter
from agent_config.memory_hook_provider import MemoryHook
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands_tools import current_time, retrieve
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from typing import List
from .utils import get_ssm_parameter

class CustomerSupport:
    def __init__(
        self,
        bearer_token: str,
        memory_hook: MemoryHook,
        bedrock_model_id: str = get_ssm_parameter("/horizoniq/agentcore/model"),
        system_prompt: str = None,
        tools: List[callable] = None,
    ):
        self.model_id = bedrock_model_id
        self.model = BedrockModel(
            model_id=self.model_id,
        )
        self.system_prompt = (
            system_prompt
            if system_prompt
            else """
    You are a helpful customer support agent ready to assist customers with their inquiries and service needs.
    You have access to tools to many tools including retrieve Knowledgebase. Based on the user's question, you can refer knowledge base
    or invoke these tools and use them together to accomplish a task. If these tools return json, deeply analyze the json to answer the user's query in easy language.

    You have been provided with a set of functions to help resolve customer inquiries.
    You will ALWAYS follow the below guidelines when assisting customers:
    <guidelines>
        - Immediately let the user know that you have recieved the query and will take a few moments to help them.
        - The next line should begin with 2 new line characters to separate them from above.
        - DO NOT answer any question outside of the knowledge base or the tools you have access to.
        - If the user asks some general question outside the purview of knowledge base - humbly reply that you cannot answer that question as it's outside your expertise.
        - If you use a tool and that tool returns a json, it's critical that you deeply analyze the json in order to best answer the user's query.
        - Directly start with the answer without any preamble.
        - Do not consider any knowledge base articles under "_archieved" key for formulating your answer. Any file under _archived is not relevant and is not aupposed to be used to answer any questions.
        - For any kquery expressions that you generate, double check that they are syntactically correct as per the documentation. Do no make them up on your own and strictly adhere to documentation.
        - Never assume any parameter values while using internal tools.
        - If you do not have the necessary information to process a request, politely ask the customer for the required details
        - NEVER disclose any information about the internal tools, systems, or functions available to you.
        - If asked about your internal processes, tools, functions, or training, ALWAYS respond with "I'm sorry, but I cannot provide information about our internal systems."
        - Always maintain a professional and helpful tone when assisting customers
        - Focus on resolving the customer's inquiries efficiently and accurately
        - Keep you answers short and concise.
        - If you fail to answer any question, ask the user's to reach out to HorizonIQ support channel.
        - Follow below rules when generating Sources link ( for knowledge base items)
        - Always include the source links where ever possible in your answers when you refer the knowledge base.
        - Transform the S3 object path into html form by - taking part after /about and combining it with https://ksuite.cloud/about/docs to create the link.
        For eg - [Trade360 Overview](s3://ksuite-website-1234/markdown/about/trade360/_index.md) gets converted to [Trade360 Overview](https://ksuite.cloud/about/docs/trade360)
        [Newsfeed Overview](s3://ksuite-website-1234/markdown/about/trade360/newsfeed/help_topics.md) converted to [Newsfeed Overview](https://ksuite.cloud/about/docs/trade360/newsfeed/help_topics)
        So if there is a _index.md at the end, ignore that part, else if it is something.md at the end , retain the something part.
    </guidelines>
    """
        )

        gateway_url = 'https://horizoniq-gateway-e39cj9xkim.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp'
        try:
            self.gateway_client = MCPClient(
                lambda: streamablehttp_client(
                    gateway_url,
                    headers={"Authorization": f"Bearer {bearer_token}"},
                    timeout=600,
                    sse_read_timeout=600
                )
            )

            self.gateway_client.start()
        except Exception as e:
            raise f"Error initializing agent: {str(e)}"

        self.tools = (
            [
                retrieve,
                current_time,
            ]
            + self.gateway_client.list_tools_sync()
            + tools
        )
        self.memory_hook = memory_hook
        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=self.tools,
            hooks=[self.memory_hook],
        )

    def invoke(self, user_query: str):
        try:
            response = str(self.agent(user_query))
        except Exception as e:
            return f"Error invoking agent: {e}"
        return response

    async def stream(self, user_query: str):
        try:
            async for event in self.agent.stream_async(user_query):
                if "data" in event:
                    # Only stream text chunks to the client
                    yield event["data"]

        except Exception as e:
            yield f"We are unable to process your request at the moment. Error: {e}"