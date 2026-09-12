"""Main coding agent orchestration.

The CodingAgent provides intelligent code assistance by:
- Searching indexed documentation
- Querying knowledge graphs for relationships
- Maintaining session memory for context
- Analyzing code structure

LLM-agnostic: Works with Ollama (local), OpenAI, or Anthropic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from docugraph.core.llm import LLMProviderBase, get_llm_provider


class AgentAction(Enum):
    """Actions the agent can take."""

    SEARCH_DOCS = "search_docs"
    SEARCH_CODE = "search_code"
    GRAPH_QUERY = "graph_query"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    RESPOND = "respond"


@dataclass
class AgentContext:
    """Context for agent execution."""

    repository: str = "default"
    branch: str = "main"
    session_id: str | None = None
    working_directory: str | None = None
    language: str | None = None  # Primary programming language
    framework: str | None = None  # Primary framework
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """A message in the agent conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tool_name: str | None = None
    tool_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    data: Any
    error: str | None = None
    took_ms: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the coding agent."""

    # LLM settings
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000

    # Search settings
    search_top_k: int = 5
    code_search_top_k: int = 10

    # Memory settings
    max_context_messages: int = 20
    include_memory_context: bool = True

    # Behavior
    auto_search_threshold: float = 0.7  # Confidence to auto-search
    verbose: bool = False


class CodingAgent:
    """Intelligent coding assistant with RAG capabilities.

    Combines multiple retrieval methods to provide contextual assistance:
    - Vector search over indexed documentation
    - Knowledge graph queries for relationships
    - Session memory for conversation context
    - Code analysis for structure understanding

    Example:
        agent = CodingAgent()
        response = await agent.query("How do I use asyncio.gather?")
        print(response)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        context: AgentContext | None = None,
    ) -> None:
        """Initialize the coding agent.

        Args:
            config: Agent configuration
            context: Initial execution context
        """
        self._config = config or AgentConfig()
        self._context = context or AgentContext()
        self._messages: list[AgentMessage] = []
        self._llm: LLMProviderBase | None = None
        self._tools: dict[str, Callable[..., Any]] = {}

        # Register default tools
        self._register_default_tools()

    def _get_llm(self) -> LLMProviderBase:
        """Lazy load LLM provider."""
        if self._llm is None:
            self._llm = get_llm_provider()
        return self._llm

    def _register_default_tools(self) -> None:
        """Register the default tools."""
        # Import via explicit submodule paths — the tools package re-exports
        # same-named functions that shadow the module names.
        from docugraph.agents.tools.graph_query import graph_query
        from docugraph.agents.tools.memory_ops import memory_recall, memory_store
        from docugraph.agents.tools.search_code import search_code
        from docugraph.agents.tools.search_docs import search_docs

        self._tools = {
            "search_docs": search_docs,
            "search_code": search_code,
            "graph_query": graph_query,
            "memory_store": memory_store,
            "memory_recall": memory_recall,
        }

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a custom tool.

        Args:
            name: Tool name
            func: Tool function (sync or async)
        """
        self._tools[name] = func

    @property
    def context(self) -> AgentContext:
        """Get current execution context."""
        return self._context

    @context.setter
    def context(self, ctx: AgentContext) -> None:
        """Set execution context."""
        self._context = ctx

    @property
    def messages(self) -> list[AgentMessage]:
        """Get conversation messages."""
        return self._messages.copy()

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **kwargs: Additional message metadata
        """
        self._messages.append(
            AgentMessage(
                role=role,
                content=content,
                **kwargs,
            )
        )

        # Trim old messages if needed
        if len(self._messages) > self._config.max_context_messages:
            # Keep system messages and recent messages
            system_msgs = [m for m in self._messages if m.role == "system"]
            other_msgs = [m for m in self._messages if m.role != "system"]
            keep_count = self._config.max_context_messages - len(system_msgs)
            self._messages = system_msgs + other_msgs[-keep_count:]

    def clear_messages(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        import time

        if tool_name not in self._tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
            )

        start_time = time.time()
        tool_func = self._tools[tool_name]

        try:
            # Add context to arguments
            arguments["_context"] = self._context

            # Check if async
            import asyncio

            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**arguments)
            else:
                result = tool_func(**arguments)

            elapsed_ms = (time.time() - start_time) * 1000

            return ToolResult(
                success=True,
                data=result,
                took_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                took_ms=elapsed_ms,
            )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tools_desc = []
        for name in self._tools:
            if name == "search_docs":
                tools_desc.append("- search_docs(query, top_k): Search indexed documentation")
            elif name == "search_code":
                tools_desc.append("- search_code(query, language, top_k): Search for code examples")
            elif name == "graph_query":
                tools_desc.append("- graph_query(query): Query knowledge graph for relationships")
            elif name == "memory_store":
                tools_desc.append("- memory_store(key, value): Store information for later")
            elif name == "memory_recall":
                tools_desc.append("- memory_recall(key): Retrieve stored information")

        return f"""You are an intelligent coding assistant with access to indexed documentation and a knowledge graph.

Your capabilities:
{chr(10).join(tools_desc)}

Context:
- Repository: {self._context.repository}
- Branch: {self._context.branch}
- Language: {self._context.language or "Not specified"}
- Framework: {self._context.framework or "Not specified"}

Guidelines:
1. Search documentation before answering technical questions
2. Use the knowledge graph for understanding relationships between concepts
3. Store important decisions and context in memory for future reference
4. Provide clear, accurate answers with references to sources
5. If unsure, acknowledge uncertainty and suggest where to find more information

When you need to use a tool, respond with a JSON object:
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}

After receiving tool results, synthesize the information into a helpful response."""

    async def _decide_action(self, user_input: str) -> tuple[AgentAction, dict[str, Any]]:
        """Decide what action to take based on user input.

        Args:
            user_input: User's message

        Returns:
            Tuple of (action, arguments)
        """
        # Simple heuristics for action selection
        user_lower = user_input.lower()

        # Memory-related queries
        if any(kw in user_lower for kw in ["remember", "recall", "what did"]):
            return AgentAction.MEMORY_RECALL, {"key": "*"}

        # Explicit search requests
        if any(kw in user_lower for kw in ["search", "find", "look up", "how to", "what is"]):
            return AgentAction.SEARCH_DOCS, {"query": user_input}

        # Code-related queries
        if any(kw in user_lower for kw in ["code example", "show code", "implement", "function"]):
            return AgentAction.SEARCH_CODE, {"query": user_input}

        # Relationship queries
        if any(kw in user_lower for kw in ["related to", "depends on", "uses", "relationship"]):
            return AgentAction.GRAPH_QUERY, {"query": user_input}

        # Default: Search docs for technical questions
        if "?" in user_input or any(kw in user_lower for kw in ["explain", "describe", "tell me"]):
            return AgentAction.SEARCH_DOCS, {"query": user_input}

        # Otherwise, respond directly
        return AgentAction.RESPOND, {}

    async def query(
        self,
        user_input: str,
        auto_search: bool = True,
    ) -> str:
        """Query the agent with a user message.

        Args:
            user_input: User's question or request
            auto_search: Automatically search if relevant

        Returns:
            Agent's response
        """
        # Add user message
        self.add_message("user", user_input)

        # Decide action
        action, action_args = await self._decide_action(user_input)

        tool_context = ""

        if action != AgentAction.RESPOND and auto_search:
            # Execute the relevant tool
            tool_name = action.value
            result = await self.execute_tool(tool_name, action_args)

            if result.success and result.data:
                tool_context = self._format_tool_result(tool_name, result.data)

                # Add tool message
                self.add_message(
                    "tool",
                    tool_context,
                    tool_name=tool_name,
                    tool_result=result.data,
                )

        # Build prompt for LLM
        system_prompt = self._build_system_prompt()

        # Build conversation context
        messages_context = []
        for msg in self._messages[-self._config.max_context_messages :]:
            if msg.role == "user":
                messages_context.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                messages_context.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                messages_context.append(f"[Tool Result: {msg.tool_name}]\n{msg.content}")

        full_prompt = f"""Based on the conversation and any tool results, provide a helpful response.

{chr(10).join(messages_context)}

Provide a clear, helpful response to the user's question."""

        try:
            llm = self._get_llm()
            response = await llm.generate(
                prompt=full_prompt,
                system_prompt=system_prompt,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )

            # Check if LLM wants to use a tool
            response = response.strip()
            if response.startswith("{") and '"tool"' in response:
                try:
                    tool_call = json.loads(response)
                    if "tool" in tool_call and "arguments" in tool_call:
                        tool_result = await self.execute_tool(
                            tool_call["tool"],
                            tool_call["arguments"],
                        )
                        if tool_result.success:
                            # Add tool result and re-query
                            tool_context = self._format_tool_result(
                                tool_call["tool"],
                                tool_result.data,
                            )
                            self.add_message(
                                "tool",
                                tool_context,
                                tool_name=tool_call["tool"],
                                tool_result=tool_result.data,
                            )
                            # Get final response
                            response = await llm.generate(
                                prompt=f"{full_prompt}\n\n[Tool Result]\n{tool_context}\n\nNow provide the final response:",
                                system_prompt=system_prompt,
                                temperature=self._config.llm_temperature,
                                max_tokens=self._config.llm_max_tokens,
                            )
                except json.JSONDecodeError:
                    pass  # Not a tool call

            # Add assistant response
            self.add_message("assistant", response)

            return response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.add_message("assistant", error_msg)
            return error_msg

    def _format_tool_result(self, tool_name: str, data: Any) -> str:
        """Format tool result for context.

        Args:
            tool_name: Name of the tool
            data: Tool result data

        Returns:
            Formatted string
        """
        if isinstance(data, list):
            if tool_name == "search_docs":
                # Format search results
                parts = []
                for i, item in enumerate(data[:5], 1):
                    if isinstance(item, dict):
                        content = item.get("content", "")[:500]
                        source = item.get("source", "Unknown")
                        score = item.get("score", 0)
                        parts.append(f"[{i}] (score: {score:.2f}) {source}\n{content}")
                    else:
                        parts.append(f"[{i}] {str(item)[:500]}")
                return "\n\n".join(parts)

            elif tool_name == "search_code":
                # Format code results
                parts = []
                for i, item in enumerate(data[:5], 1):
                    if isinstance(item, dict):
                        code = item.get("code", "")[:1000]
                        path = item.get("path", "Unknown")
                        lang = item.get("language", "")
                        parts.append(f"[{i}] {path}\n```{lang}\n{code}\n```")
                    else:
                        parts.append(f"[{i}] {str(item)[:500]}")
                return "\n\n".join(parts)

            elif tool_name == "graph_query":
                # Format graph results
                parts = []
                for item in data[:10]:
                    if isinstance(item, dict):
                        fact = item.get("fact", str(item))
                        parts.append(f"- {fact}")
                    else:
                        parts.append(f"- {str(item)}")
                return "\n".join(parts)

            else:
                return json.dumps(data, indent=2, default=str)[:2000]

        elif isinstance(data, dict):
            return json.dumps(data, indent=2, default=str)[:2000]

        else:
            return str(data)[:2000]

    async def summarize_session(self) -> str:
        """Summarize the current session.

        Returns:
            Summary of the conversation
        """
        if not self._messages:
            return "No messages in session."

        user_questions = [m.content for m in self._messages if m.role == "user"]
        tool_calls = [m.tool_name for m in self._messages if m.tool_name]

        summary_parts = [
            f"Session with {len(self._messages)} messages",
            f"Repository: {self._context.repository}/{self._context.branch}",
            f"Questions asked: {len(user_questions)}",
            f"Tools used: {', '.join(set(tool_calls)) or 'None'}",
        ]

        return "\n".join(summary_parts)


def get_coding_agent(
    config: AgentConfig | None = None,
    context: AgentContext | None = None,
) -> CodingAgent:
    """Factory function to get a CodingAgent instance.

    Args:
        config: Optional agent configuration
        context: Optional execution context

    Returns:
        CodingAgent instance
    """
    return CodingAgent(config=config, context=context)
