---
name: context
description: Build comprehensive context from docs, memory, and architecture
context: fork
allowed-tools: Read
---

# Build Context

Assemble comprehensive context by combining documentation search, stored memory, and architectural knowledge. Use this when you need rich background information for a complex task.

## Usage

```
/context implementing user authentication
/context setting up the database layer
/context understanding the payment flow
/context refactoring the API endpoints
```

## Instructions

1. **Extract the topic** from $ARGUMENTS
   - This describes what you need context about

2. **Gather documentation context:**
   - Use `hybrid_search` MCP tool with the topic as query
   - Set `top_k: 5` for focused results
   - Set `include_graph: true` to get related entities
   - Set `mode: "all"` for comprehensive search

3. **Gather memory context:**
   - Use `memory_recall` MCP tool with `list_all: true`
   - Filter results relevant to the topic
   - Include key decisions and preferences

4. **Gather architectural context:**
   - Use `query` MCP tool with `queryType: "entities"` for components
   - Use `query` MCP tool with `queryType: "governance"` for rules and decisions
   - Include relevant components, their dependencies, and applicable rules

5. **Synthesize and present:**
   - Organize context into clear sections
   - Highlight the most relevant information first
   - Note any gaps in available context
   - Suggest what additional documentation might help

## Output Format

```
## Context: [Topic]

### Relevant Documentation
[Top search results with key excerpts]

### Stored Decisions & Preferences
[Relevant items from memory]

### Architecture
**Related Components:**
- [Component]: [Description and dependencies]

**Applicable Rules:**
- [Rule]: [Content]

**Related Decisions:**
- [ADR]: [Summary]

### Summary
[Brief synthesis of the most important context]

### Gaps
[Any missing context that might be helpful to index]
```

## Example

**Input:** `/context implementing user authentication`

**Output:**
```
## Context: Implementing User Authentication

### Relevant Documentation
1. **FastAPI Security** (score: 0.92)
   > OAuth2 with Password flow using JWT tokens...

2. **JWT Best Practices** (score: 0.85)
   > Always use httpOnly cookies for refresh tokens...

### Stored Decisions & Preferences
- **auth**: JWT with httpOnly cookies, refresh tokens stored server-side
- **db**: PostgreSQL 15 with Prisma ORM

### Architecture
**Related Components:**
- **auth-service**: Handles authentication, depends on database
- **api-gateway**: Routes requests, depends on auth-service

**Applicable Rules:**
- **no-direct-db**: All database access must go through repository layer

**Related Decisions:**
- **adr-001**: Use JWT for Authentication - Chose for horizontal scaling

### Summary
Authentication should use JWT tokens with OAuth2 Password flow.
Refresh tokens go in httpOnly cookies. All DB access through repositories.
The auth-service component handles this, called from api-gateway.

### Gaps
Consider indexing your framework's security documentation if not already done.
```

## When to Use

- Starting work on a new feature
- Onboarding to an unfamiliar part of the codebase
- Before making architectural changes
- When you need to understand existing decisions
