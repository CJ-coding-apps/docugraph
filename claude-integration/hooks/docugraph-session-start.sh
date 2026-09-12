#!/bin/bash
# DocuGraph Session Start Hook
# Loads stored context from DocuGraph memory when a Claude Code session starts
#
# This hook runs at SessionStart and provides any stored context as
# additional context for the conversation.

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)

# Get repository and branch context
REPO=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename 2>/dev/null || echo "")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# If not in a git repo, skip context loading
if [ -z "$REPO" ]; then
    echo '{"continue": true}'
    exit 0
fi

# Try to load stored context from DocuGraph
CONTEXT=""
if command -v docugraph &> /dev/null; then
    CONTEXT=$(docugraph memory show --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "")
fi

# If we have context, include it in the response
if [ -n "$CONTEXT" ] && [ "$CONTEXT" != "{}" ] && [ "$CONTEXT" != "No memories found." ]; then
    # Escape the context for JSON
    ESCAPED_CONTEXT=$(echo "$CONTEXT" | jq -Rs '.')

    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## DocuGraph Session Context\\n\\nLoaded from memory for **${REPO}/${BRANCH}**:\\n\\n${CONTEXT}\\n\\nUse \`/recall\` to see all stored context or \`/remember key: value\` to store new context."
  }
}
EOF
else
    echo '{"continue": true}'
fi
