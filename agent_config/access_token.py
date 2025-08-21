from .utils import get_ssm_parameter
from bedrock_agentcore.identity.auth import requires_access_token


@requires_access_token(
    provider_name=get_ssm_parameter("/horizoniq/agentcore/identity/client"),
    scopes=[],
    auth_flow="M2M",
)
async def get_gateway_access_token(access_token: str):
    return access_token