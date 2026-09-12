#!/bin/bash
# DocuGraph Session End Hook
# Saves uncompleted todos and active plans before session ends
#
# Triggers: Stop event
# Runs BEFORE the existing prompt hook
#
# Input: session_id, transcript_path

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')

# Get repository context
REPO=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename 2>/dev/null || echo "default")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# Skip if docugraph is not installed
if ! command -v docugraph &> /dev/null; then
    echo '{"continue": true}'
    exit 0
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVED_ITEMS=""
ITEMS_SAVED=0

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then

    # Extract uncompleted todos (last TodoWrite state)
    TODOS=$(cat "$TRANSCRIPT_PATH" | jq -s '
        [.[] | select(.tool_name == "TodoWrite")] | last | .tool_input.todos // []
        | map(select(.status == "pending" or .status == "in_progress"))
    ' 2>/dev/null || echo "[]")

    if [ "$TODOS" != "[]" ] && [ "$TODOS" != "null" ] && [ -n "$TODOS" ]; then
        TODO_COUNT=$(echo "$TODOS" | jq 'length' 2>/dev/null || echo "0")
        if [ "$TODO_COUNT" -gt 0 ]; then
            docugraph memory set "_session_todos_${TIMESTAMP}" "$TODOS" \
                --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true
            SAVED_ITEMS="${SAVED_ITEMS}- **${TODO_COUNT} uncompleted task(s)**\n"
            ITEMS_SAVED=$((ITEMS_SAVED + 1))
        fi
    fi

    # Extract active plan content (from writes to .claude/plans/)
    PLAN_CONTENT=$(cat "$TRANSCRIPT_PATH" | jq -rs '
        [.[] | select(
            (.tool_name == "Write" or .tool_name == "Edit") and
            (.tool_input.file_path | test("\\.claude/plans/"))
        )] | last | .tool_input.content // .tool_input.new_string // empty
    ' 2>/dev/null || echo "")

    PLAN_PATH=$(cat "$TRANSCRIPT_PATH" | jq -rs '
        [.[] | select(
            (.tool_name == "Write" or .tool_name == "Edit") and
            (.tool_input.file_path | test("\\.claude/plans/"))
        )] | last | .tool_input.file_path // empty
    ' 2>/dev/null || echo "")

    if [ -n "$PLAN_CONTENT" ]; then
        PLAN_DATA=$(jq -n --arg path "$PLAN_PATH" --arg content "$PLAN_CONTENT" \
            '{path: $path, content: $content}')
        docugraph memory set "_session_plan_${TIMESTAMP}" "$PLAN_DATA" \
            --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true
        SAVED_ITEMS="${SAVED_ITEMS}- **Active plan:** \`${PLAN_PATH}\`\n"
        ITEMS_SAVED=$((ITEMS_SAVED + 1))
    fi
fi

# Output response
if [ "$ITEMS_SAVED" -gt 0 ]; then
    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## Session Context Saved\n\nThe following has been saved to DocuGraph for your next session:\n\n${SAVED_ITEMS}\nUse \`/recall\` in your next session to restore this context."
  }
}
EOF
else
    echo '{"continue": true}'
fi
