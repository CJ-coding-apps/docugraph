"""Agent orchestration and tool implementations.

The agents module provides:
- CodingAgent: Main agent for code assistance with RAG
- Tools: Modular tools for search, graph queries, and memory
"""

from docugraph.agents.coding_agent import (
    AgentAction,
    AgentConfig,
    AgentContext,
    AgentMessage,
    CodingAgent,
    ToolResult,
    get_coding_agent,
)

__all__ = [
    # Agent
    "CodingAgent",
    "AgentConfig",
    "AgentContext",
    "AgentMessage",
    "AgentAction",
    "ToolResult",
    "get_coding_agent",
]
