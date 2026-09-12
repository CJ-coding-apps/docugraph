---
name: remember
description: Store important context and decisions for future sessions
allowed-tools: Read, Bash
---

# Remember Context

Store important context, decisions, and preferences in DocuGraph's memory system. This information persists across sessions and is scoped by repository and branch.

## Usage

```
/remember auth: We're using JWT tokens with refresh token rotation
/remember db: PostgreSQL 15 with Prisma ORM
/remember patterns: Repository pattern for data access, Service layer for business logic
/remember decision: Chose React over Vue for better TypeScript support
/remember todo: Need to implement rate limiting before launch
```

## Instructions

1. **Parse the input** from $ARGUMENTS:
   - Format is `key: value` or `key = value`
   - The key is everything before the first `:` or `=`
   - The value is everything after (trimmed)
   - If no delimiter, treat entire input as value with key "note"

2. **Detect repository context:**
   - Get repository name: !`git rev-parse --show-toplevel 2>/dev/null | xargs basename || echo "default"`!
   - Get current branch: !`git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"`!

3. **Store in memory:**
   - Use the `memory_store` MCP tool from docugraph
   - Set `key` to the parsed key
   - Set `value` to the parsed value
   - Set `repository` to the detected repo name
   - Set `branch` to the detected branch

4. **Confirm storage:**
   - Echo back what was stored
   - Show the repository/branch scope
   - Mention that this will be available in future sessions

## Examples

**Store an architectural decision:**
```
/remember auth: JWT with httpOnly cookies, refresh tokens stored server-side
```
Output: `Stored "auth" for myproject/main`

**Store a reminder:**
```
/remember todo: Add input validation to all API endpoints
```
Output: `Stored "todo" for myproject/feature-branch`

**Store with equals sign:**
```
/remember db_host = localhost:5432
```
Output: `Stored "db_host" for myproject/main`

## Related Skills

- `/recall` - Retrieve stored context
- `/architecture decision` - Track formal architectural decisions (ADRs)
