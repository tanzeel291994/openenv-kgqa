"""
KGQA Environment Client.

Connects to a KGQA environment server via the openenv MCPToolClient.

Example:
    >>> from client import KGQAEnv
    >>> async with KGQAEnv(base_url="http://localhost:7860") as env:
    ...     await env.reset(task_type="triple_completion")
    ...     tools = await env.list_tools()
    ...     result = await env.call_tool("get_task_info")
"""

from openenv.core.mcp_client import MCPToolClient


class KGQAEnv(MCPToolClient):
    """Client for the KGQA environment. MCPToolClient provides all functionality."""

    pass
