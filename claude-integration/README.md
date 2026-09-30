# DocuGraph AI - Claude Code Integration

This directory contains skills and hooks that integrate DocuGraph AI with Claude Code for a seamless documentation-aware coding experience.

## Features

### Skills (Slash Commands)

| Skill | Description |
|-------|-------------|
| `/index` | Index documentation from URLs, git repos, or local files |
| `/docs` | Search indexed documentation with natural language |
| `/remember` | Store context and decisions for future sessions |
| `/recall` | Retrieve stored context from memory |
| `/architecture` | Manage components, ADRs, and governance rules |
| `/context` | Build comprehensive context from all sources |

### Hooks (Automatic Triggers)

| Hook | Event | Description |
|------|-------|-------------|
| Session Start | `SessionStart` | Loads stored context when starting a session |
| Pre-Compact | `PreCompact` | Saves full context to DocuGraph before compression |
| Post-Compact | `SessionStart` (compact) | Restores context after compression completes |
| Auto-Search | `UserPromptSubmit` | Searches docs for relevant context on each prompt |
| Rule Check | `PreToolUse` (Edit/Write) | Warns about architecture rule violations |
| Auto-Index | `PostToolUse` (Edit/Write) | Re-indexes documentation files when modified |
| Context Save | `Stop` | Prompts to save important decisions at session end |

### Context Compression Flow

During long coding sessions, Claude Code's context window may fill up, triggering automatic compression. The Pre-Compact and Post-Compact hooks work together to preserve important context:

```
Long coding session fills context window
              ↓
┌─────────────────────────────────────┐
│         PreCompact fires            │
│  • Access full transcript           │
│  • Extract modified files list      │
│  • Save session state to DocuGraph  │
│  • Store entities in graph memory   │
└─────────────────────────────────────┘
              ↓
     Claude Code compresses context
              ↓
┌─────────────────────────────────────┐
│  SessionStart (compact) fires       │
│  • Load stored user context         │
│  • Restore recent file list         │
│  • Reload active components/rules   │
│  • Inject as additionalContext      │
└─────────────────────────────────────┘
              ↓
    Continue coding with restored context
```

This prevents common issues during long sessions:
- Forgetting which files were modified
- Losing track of architectural decisions
- Repeating work that was already done
- Breaking consistency with earlier changes

## Prerequisites

1. **DocuGraph AI installed and configured** (not yet published to PyPI, so
   install from a checkout of this repository):
   ```bash
   pip install -e .
   # or
   uv sync
   ```

2. **DocuGraph MCP server configured** in `~/.claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "docugraph": {
         "command": "docugraph",
         "args": ["mcp-server"]
       }
     }
   }
   ```

3. **jq installed** (for hook scripts):
   ```bash
   # macOS
   brew install jq

   # Ubuntu/Debian
   sudo apt-get install jq
   ```

## Installation

### Option 1: Project-Level Installation (Recommended)

Install for a specific project:

```bash
# Navigate to your project
cd /path/to/your/project

# Create Claude Code directories
mkdir -p .claude/skills .claude/hooks

# Copy skills
cp -r /path/to/docugraph/claude-integration/skills/* .claude/skills/

# Copy hooks
cp /path/to/docugraph/claude-integration/hooks/* .claude/hooks/
chmod +x .claude/hooks/*.sh

# Copy settings (or merge with existing)
cp /path/to/docugraph/claude-integration/settings.json .claude/settings.json
```

### Option 2: Personal Installation (All Projects)

Install for all your projects:

```bash
# Copy skills to personal location
mkdir -p ~/.claude/skills
cp -r /path/to/docugraph/claude-integration/skills/* ~/.claude/skills/

# For hooks, you'll need to use absolute paths or install per-project
```

### Option 3: One-Line Install Script

```bash
# From the docugraph repository
./claude-integration/install.sh /path/to/your/project
```

## Usage Examples

### Index Documentation

```bash
# Index a documentation website
/index https://fastapi.tiangolo.com/tutorial/

# Index a GitHub repository
/index git:https://github.com/tiangolo/fastapi

# Index local documentation
/index ./docs
```

### Search Documentation

```bash
# Natural language search
/docs how to handle authentication errors

# Search with more results
/docs database connection pooling --top 10
```

### Manage Session Context

```bash
# Store important decisions
/remember auth: Using JWT with httpOnly cookies for refresh tokens
/remember db: PostgreSQL 15 with Prisma ORM
/remember patterns: Repository pattern for data access

# Recall stored context
/recall auth
/recall *  # List all
```

### Track Architecture

```bash
# Create components
/architecture component create api-gateway "API Gateway" --kind service
/architecture component create auth-service "Auth Service" --depends api-gateway

# Create decisions (ADRs)
/architecture decision create adr-001 "Use JWT" --rationale "Need stateless auth for scaling"

# Create governance rules
/architecture rule create no-direct-db "No Direct DB" --content "Use repository pattern"

# Analyze architecture
/architecture cycles    # Find circular dependencies
/architecture pagerank  # Find most important components
```

### Build Rich Context

```bash
# Get comprehensive context for a task
/context implementing user authentication
```

## Configuration

### Customizing Hooks

Edit `.claude/settings.json` to:

- **Disable a hook:** Remove its entry from the settings
- **Adjust timeouts:** Change the `timeout` value (in seconds)
- **Add matchers:** Restrict hooks to specific tools

Example - disable auto-search:
```json
{
  "hooks": {
    "SessionStart": [...],
    "PreCompact": [...],
    // "UserPromptSubmit": [...],  // Commented out = disabled
    "PreToolUse": [...],
    "PostToolUse": [...],
    "Stop": [...]
  }
}
```

### Adding Custom Rules

Create architecture rules that the rule-check hook will enforce:

```bash
# Example rules
/architecture rule create no-console "No Console Logs" --content "Use logger instead of console.log"
/architecture rule create no-any "No TypeScript Any" --content "Avoid using 'any' type"
/architecture rule create require-error-handling "Error Handling Required" --content "Async functions must have try/catch"
```

## File Structure

```
claude-integration/
├── skills/
│   ├── index/
│   │   └── SKILL.md       # /index command
│   ├── docs/
│   │   └── SKILL.md       # /docs command
│   ├── remember/
│   │   └── SKILL.md       # /remember command
│   ├── recall/
│   │   └── SKILL.md       # /recall command
│   ├── architecture/
│   │   └── SKILL.md       # /architecture command
│   └── context/
│       └── SKILL.md       # /context command
├── hooks/
│   ├── docugraph-session-start.sh   # Load context on session start
│   ├── docugraph-pre-compact.sh     # Save context before compression
│   ├── docugraph-post-compact.sh    # Restore context after compression
│   ├── docugraph-auto-search.sh     # Auto-search on prompts
│   ├── docugraph-rule-check.sh      # Check rules before edits
│   └── docugraph-auto-index.sh      # Re-index docs on change
├── settings.json                     # Hook configuration
└── README.md                         # This file
```

## Troubleshooting

### Skills not appearing

1. Check skills are in the correct location:
   - Project: `.claude/skills/<skill-name>/SKILL.md`
   - Personal: `~/.claude/skills/<skill-name>/SKILL.md`

2. Restart Claude Code to reload skills

### Hooks not running

1. Check hook scripts are executable:
   ```bash
   chmod +x .claude/hooks/*.sh
   ```

2. Verify `jq` is installed:
   ```bash
   which jq
   ```

3. Test hooks manually:
   ```bash
   echo '{"prompt": "test query"}' | .claude/hooks/docugraph-auto-search.sh
   ```

### DocuGraph commands not found

1. Ensure DocuGraph is installed:
   ```bash
   docugraph --version
   ```

2. Check it's in your PATH:
   ```bash
   which docugraph
   ```

## Contributing

To add new skills or hooks:

1. Create a new skill directory with `SKILL.md`
2. Or add a new hook script and update `settings.json`
3. Test thoroughly before committing
4. Update this README with usage instructions

## License

Same license as DocuGraph AI.
