#!/bin/bash
# DocuGraph Pre-Compact Hook
# Saves context to DocuGraph before context compression
#
# This hook runs BEFORE compression (manual /compact or automatic),
# with access to the full transcript. It extracts important information
# and saves it to DocuGraph's memory so it can be restored after compression.
#
# Triggers: PreCompact (manual or auto)
# Input: session_id, trigger, transcript_path, custom_instructions

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
TRIGGER=$(echo "$INPUT" | jq -r '.trigger // "auto"')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')
CUSTOM_INSTRUCTIONS=$(echo "$INPUT" | jq -r '.custom_instructions // empty')

# Get repository context
REPO=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename 2>/dev/null || echo "default")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# Skip if docugraph is not installed
if ! command -v docugraph &> /dev/null; then
    echo '{"continue": true}'
    exit 0
fi

# Timestamp for this snapshot
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Initialize variables
MODIFIED_FILES=""
FILE_COUNT=0
CONTEXT_SAVED=false

# Extract important context from transcript before compression
if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then

    # 1. Extract files modified in this session (from Edit and Write tool calls)
    MODIFIED_FILES=$(cat "$TRANSCRIPT_PATH" | jq -r '
        select(.tool_name == "Edit" or .tool_name == "Write")
        | .tool_input.file_path // empty' 2>/dev/null | sort -u | grep -v '^$' | head -30 || echo "")

    if [ -n "$MODIFIED_FILES" ]; then
        FILE_COUNT=$(echo "$MODIFIED_FILES" | wc -l | tr -d ' ')
    fi

    # 2. Extract any errors encountered (from tool failures)
    ERRORS=$(cat "$TRANSCRIPT_PATH" | jq -r '
        select(.tool_name and .isError == true)
        | "\(.tool_name): \(.content[0].text // "error")"' 2>/dev/null | tail -5 || echo "")

    # 3. Extract bash commands that were run
    COMMANDS=$(cat "$TRANSCRIPT_PATH" | jq -r '
        select(.tool_name == "Bash")
        | .tool_input.command // empty' 2>/dev/null | tail -10 || echo "")

    # 4. Extract uncompleted todos from TodoWrite calls
    # Get the LAST TodoWrite call (most recent state)
    TODOS=$(cat "$TRANSCRIPT_PATH" | jq -s '
        [.[] | select(.tool_name == "TodoWrite")] | last | .tool_input.todos // []
        | map(select(.status == "pending" or .status == "in_progress"))
    ' 2>/dev/null || echo "[]")

    # 5. Extract active plan content (from writes to .claude/plans/)
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

    # Save modified files list to memory
    if [ -n "$MODIFIED_FILES" ]; then
        # Format as a clean list
        FILES_JSON=$(echo "$MODIFIED_FILES" | jq -R -s 'split("\n") | map(select(length > 0))')
        docugraph memory set "_session_files_${TIMESTAMP}" "$FILES_JSON" \
            --repo "$REPO" --branch "$BRANCH" 2>/dev/null && CONTEXT_SAVED=true || true
    fi

    # Save session summary
    TODO_COUNT=$(echo "$TODOS" | jq 'length' 2>/dev/null || echo "0")
    HAS_PLAN="no"
    [ -n "$PLAN_CONTENT" ] && HAS_PLAN="yes"
    SUMMARY="Compression trigger: ${TRIGGER}. Files: ${FILE_COUNT}. Todos: ${TODO_COUNT}. Active plan: ${HAS_PLAN}."
    if [ -n "$CUSTOM_INSTRUCTIONS" ]; then
        SUMMARY="${SUMMARY} User note: ${CUSTOM_INSTRUCTIONS}"
    fi
    docugraph memory set "_session_summary_${TIMESTAMP}" "$SUMMARY" \
        --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true

    # Save errors if any
    if [ -n "$ERRORS" ]; then
        docugraph memory set "_session_errors_${TIMESTAMP}" "$ERRORS" \
            --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true
    fi

    # Save recent commands
    if [ -n "$COMMANDS" ]; then
        COMMANDS_JSON=$(echo "$COMMANDS" | jq -R -s 'split("\n") | map(select(length > 0))')
        docugraph memory set "_session_commands_${TIMESTAMP}" "$COMMANDS_JSON" \
            --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true
    fi

    # Save uncompleted todos
    if [ "$TODOS" != "[]" ] && [ "$TODOS" != "null" ] && [ -n "$TODOS" ]; then
        docugraph memory set "_session_todos_${TIMESTAMP}" "$TODOS" \
            --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true
    fi

    # Save active plan
    if [ -n "$PLAN_CONTENT" ]; then
        PLAN_DATA=$(jq -n --arg path "$PLAN_PATH" --arg content "$PLAN_CONTENT" \
            '{path: $path, content: $content}')
        docugraph memory set "_session_plan_${TIMESTAMP}" "$PLAN_DATA" \
            --repo "$REPO" --branch "$BRANCH" 2>/dev/null || true
    fi

    # Also store in graph memory for richer relationship querying
    if [ -n "$MODIFIED_FILES" ] && [ "$FILE_COUNT" -gt 0 ]; then
        GRAPH_CONTENT="Session snapshot before ${TRIGGER} compression at ${TIMESTAMP}. Modified ${FILE_COUNT} files: $(echo "$MODIFIED_FILES" | tr '\n' ', ' | sed 's/,$//')"
        docugraph graph add "$GRAPH_CONTENT" \
            --name "session_${TIMESTAMP}" --group "$REPO" 2>/dev/null || true
    fi
fi

# Build response
if [ "$CONTEXT_SAVED" = true ]; then
    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## Pre-Compression Context Saved\n\nBefore compression, the following was saved to DocuGraph:\n- **Trigger:** ${TRIGGER}\n- **Files modified:** ${FILE_COUNT}\n- **Uncompleted todos:** ${TODO_COUNT}\n- **Active plan:** ${HAS_PLAN}\n- **Snapshot ID:** ${TIMESTAMP}\n\nThis context will be restored after compression completes."
  }
}
EOF
else
    # No significant context to save
    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "Pre-compression: No significant session context to save (trigger: ${TRIGGER})"
  }
}
EOF
fi
