#!/bin/bash
# DocuGraph Rule Check Hook
# Checks for architecture rule violations before file modifications
#
# This hook runs at PreToolUse for Edit|Write operations and warns
# about potential rule violations.

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)

# Extract file path and content being written
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty')

# Skip if we don't have the necessary info
if [ -z "$FILE_PATH" ]; then
    echo '{"continue": true}'
    exit 0
fi

# Skip if docugraph is not installed
if ! command -v docugraph &> /dev/null; then
    echo '{"continue": true}'
    exit 0
fi

# Get repository context
REPO=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename 2>/dev/null || echo "default")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# Get active rules (as JSON)
RULES=$(docugraph rule list --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "[]")

# Skip if no rules defined
if [ "$RULES" = "[]" ] || [ "$RULES" = "No rules found." ]; then
    echo '{"continue": true}'
    exit 0
fi

WARNINGS=""

# Check for common rule patterns
# Rule: no-direct-db - No direct database access outside repository layer
if echo "$RULES" | jq -e '.[] | select(.id == "no-direct-db" or .name | test("direct.*db|database.*access"; "i"))' >/dev/null 2>&1; then
    # Check if file is NOT in a repository/data layer
    if ! echo "$FILE_PATH" | grep -qiE '(repository|repo|dal|data-access|models?/|prisma/|db/)'; then
        # Check if content contains direct DB access patterns
        if echo "$NEW_CONTENT" | grep -qE '(\$query|\$execute|\.query\(|\.execute\(|prisma\.\$|mongoose\.|sequelize\.|knex\(|sql`|SELECT |INSERT |UPDATE |DELETE )'; then
            WARNINGS="${WARNINGS}* **no-direct-db**: This change may contain direct database access. Consider using the repository pattern.\n"
        fi
    fi
fi

# Rule: no-console - No console.log in production code
if echo "$RULES" | jq -e '.[] | select(.id == "no-console" or .name | test("console"; "i"))' >/dev/null 2>&1; then
    if ! echo "$FILE_PATH" | grep -qiE '(test|spec|\.test\.|\.spec\.|debug|dev)'; then
        if echo "$NEW_CONTENT" | grep -qE 'console\.(log|debug|info|warn|error)\('; then
            WARNINGS="${WARNINGS}* **no-console**: This change adds console statements. Consider using a proper logging library.\n"
        fi
    fi
fi

# Rule: no-any - No TypeScript 'any' type
if echo "$RULES" | jq -e '.[] | select(.id == "no-any" or .name | test("any.*type"; "i"))' >/dev/null 2>&1; then
    if echo "$FILE_PATH" | grep -qE '\.(ts|tsx)$'; then
        if echo "$NEW_CONTENT" | grep -qE ': any[^a-zA-Z]|<any>|as any'; then
            WARNINGS="${WARNINGS}* **no-any**: This change uses TypeScript 'any' type. Consider using proper types or generics.\n"
        fi
    fi
fi

# Rule: require-error-handling - Async functions must have error handling
if echo "$RULES" | jq -e '.[] | select(.id == "require-error-handling" or .name | test("error.*handl"; "i"))' >/dev/null 2>&1; then
    if echo "$NEW_CONTENT" | grep -qE 'async.*\{' && ! echo "$NEW_CONTENT" | grep -qE '(try\s*\{|\.catch\(|catch\s*\()'; then
        WARNINGS="${WARNINGS}* **require-error-handling**: Async code should include error handling (try/catch or .catch()).\n"
        fi
fi

# Output warnings if any
if [ -n "$WARNINGS" ]; then
    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## Architecture Rule Warnings\\n\\nThe following rules may be affected by this change:\\n\\n${WARNINGS}\\nThese are warnings only - the change will proceed. Use \`/architecture rule list\` to see all rules."
  }
}
EOF
else
    echo '{"continue": true}'
fi
