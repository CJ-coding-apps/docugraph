---
name: architecture
description: Manage architectural components, decisions (ADRs), and governance rules
disable-model-invocation: true
allowed-tools: Read, Bash
---

# Architecture Management

Track and manage architectural components, decisions (ADRs), and governance rules in DocuGraph's graph memory. This creates a queryable knowledge graph of your system architecture.

## Usage

### Components
```
/architecture component create api-gateway "API Gateway" --kind service
/architecture component create auth-service "Auth Service" --kind service --depends api-gateway
/architecture component list
/architecture component show api-gateway
/architecture component deps api-gateway
```

### Decisions (ADRs)
```
/architecture decision create adr-001 "Use JWT for Authentication" --context "Need stateless auth" --rationale "Scales horizontally"
/architecture decision list
/architecture decision show adr-001
```

### Rules
```
/architecture rule create no-direct-db "No Direct DB Access" --content "All database access must go through repository layer"
/architecture rule list
```

### Analysis
```
/architecture show          # Visualize all components
/architecture cycles        # Detect circular dependencies
/architecture islands       # Find disconnected component groups
/architecture pagerank      # Find most important components
```

## Instructions

Parse the subcommand and action from $ARGUMENTS:

### Component Commands

**component create <id> <name> [options]:**
- Use `entity` MCP tool with `operation: "create"`, `entityType: "component"`
- Parse `--kind` (service, library, module, etc.)
- Parse `--depends` or `-d` (can be multiple)
- Parse `--description`
- Parse `--status` (active, deprecated, planned)
- Set `repository` and `branch` from git context

**component list:**
- Use `entity` MCP tool with `operation: "list"`, `entityType: "component"`
- Format as table showing id, name, kind, status, dependencies

**component show <id>:**
- Use `entity` MCP tool with `operation: "get"`, `entityType: "component"`, `id: <id>`

**component deps <id>:**
- Use `query` MCP tool with `queryType: "dependencies"`, `componentId: <id>`
- Parse `--direction` (in, out, both) default: both
- Parse `--depth` default: 1

### Decision Commands

**decision create <id> <name> [options]:**
- Use `entity` MCP tool with `operation: "create"`, `entityType: "decision"`
- Parse `--context` for decision context
- Parse `--rationale` for reasoning
- Parse `--status` (proposed, accepted, rejected, deprecated, superseded)
- Date is auto-set to today

**decision list:**
- Use `entity` MCP tool with `operation: "list"`, `entityType: "decision"`

**decision show <id>:**
- Use `entity` MCP tool with `operation: "get"`, `entityType: "decision"`, `id: <id>`

### Rule Commands

**rule create <id> <name> --content <rule-content>:**
- Use `entity` MCP tool with `operation: "create"`, `entityType: "rule"`
- `--content` is required - the actual rule definition
- Parse `--description`, `--scope`, `--severity` (info, warning, error)

**rule list:**
- Use `entity` MCP tool with `operation: "list"`, `entityType: "rule"`

### Analysis Commands

**show:**
- Use `query` MCP tool with `queryType: "entities"` for each type
- Create ASCII diagram showing components and dependencies

**cycles:**
- Use `detect` MCP tool with `pattern: "cycles"`
- Report any circular dependencies found

**islands:**
- Use `detect` MCP tool with `pattern: "islands"`
- Report disconnected component groups

**pagerank:**
- Use `analyze` MCP tool with `algorithm: "pagerank"`
- Show ranked list of components by importance

## Repository Context

Always detect and use:
- Repository: !`git rev-parse --show-toplevel 2>/dev/null | xargs basename || echo "default"`!
- Branch: !`git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main"`!

## Example Output

**Component visualization:**
```
## Architecture: myproject/main

    ┌─────────────┐     ┌─────────────┐
    │ API Gateway │────▶│Auth Service │
    │  (service)  │     │  (service)  │
    └─────────────┘     └──────┬──────┘
          │                    │
          ▼                    ▼
    ┌─────────────┐     ┌─────────────┐
    │User Service │     │  Database   │
    │  (service)  │     │  (storage)  │
    └─────────────┘     └─────────────┘

4 components, 4 dependencies
No circular dependencies detected.
```
