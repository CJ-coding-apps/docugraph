#!/bin/bash
# DocuGraph Auto-Search Hook
# Automatically searches indexed documentation when user submits a prompt
#
# This hook runs at UserPromptSubmit and provides relevant documentation
# as additional context for the conversation.

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)

# Extract the user's prompt
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Skip if prompt is too short (likely a command or simple question)
if [ ${#PROMPT} -lt 40 ]; then
    echo '{"continue": true}'
    exit 0
fi

# Skip if docugraph is not installed
if ! command -v docugraph &> /dev/null; then
    echo '{"continue": true}'
    exit 0
fi

# Skip if this looks like a meta-question about Claude Code itself
if echo "$PROMPT" | grep -qiE '(how do (you|i)|can you|what is claude|help me understand)'; then
    echo '{"continue": true}'
    exit 0
fi

# Search for relevant documentation (brief format, top 3 results)
RESULTS=$(docugraph search "$PROMPT" --top-k 3 2>/dev/null || echo "")

# If we have results, include them
if [ -n "$RESULTS" ] && [ "$RESULTS" != "No results found." ] && ! echo "$RESULTS" | grep -q "^Error"; then
    # Truncate results if too long (keep under 2000 chars)
    if [ ${#RESULTS} -gt 2000 ]; then
        RESULTS="${RESULTS:0:2000}...

(Results truncated. Use \`/docs your query\` for full search.)"
    fi

    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## Relevant Documentation (auto-searched)\\n\\n${RESULTS}\\n\\n---\\n*Use \`/docs query\` for more targeted searches or \`/index url\` to add more documentation.*"
  }
}
EOF
else
    echo '{"continue": true}'
fi
