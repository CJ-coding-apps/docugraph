---
name: recall
description: Retrieve stored context and decisions from previous sessions
allowed-tools: Read, Bash
---

# Recall Context

Retrieve stored context, decisions, and preferences from DocuGraph's memory system. Memory is scoped by repository and branch.

## Usage

```
/recall auth              # Get specific key
/recall db                # Get database-related context
/recall *                 # List all stored context
/recall                   # Same as /recall *
/recall todo%             # Pattern match (SQL LIKE syntax)
```

## Instructions

1. **Parse the input** from $ARGUMENTS:
   - If empty or `*`, list all keys
   - If ends with `%`, use as pattern for search
   - Otherwise, treat as exact key to retrieve

2. **Detect repository context:**
   - Get repository name: !`git rev-parse --show-toplevel 2>/dev/null | xargs basename || echo "default"`!
   - Get current branch: !`git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"`!

3. **Retrieve from memory:**
   - Use the `memory_recall` MCP tool from docugraph
   - For listing all: set `list_all` to `true`
   - For specific key: set `key` to the requested key
   - For pattern: set `key` to the pattern (with `%` wildcard)
   - Set `repository` and `branch` from detected context

4. **Present results:**
   - For single key: show key = value
   - For list: show all keys with their values
   - If not found: suggest using `/remember` to store context
   - Show the repository/branch scope

## Output Format

**Single key:**
```
## Recalled: auth
**Value:** JWT with httpOnly cookies, refresh tokens stored server-side
**Scope:** myproject/main
```

**All keys:**
```
## Stored Context (myproject/main)

| Key | Value |
|-----|-------|
| auth | JWT with httpOnly cookies |
| db | PostgreSQL 15 with Prisma |
| patterns | Repository pattern |

3 items stored. Use `/remember key: value` to add more.
```

**Not found:**
```
No value found for key "auth" in myproject/main.
Use `/remember auth: your value` to store it.
```

## Related Skills

- `/remember` - Store new context
- `/architecture` - View formal architectural decisions
