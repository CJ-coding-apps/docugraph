#!/bin/bash
# DocuGraph Post-Compact Hook
# Restores context from DocuGraph after context compression
#
# This hook runs AFTER compression completes, via SessionStart with "compact" matcher.
# It retrieves saved context from DocuGraph and injects it back into Claude's context,
# preventing loss of important information during long coding sessions.
#
# Triggers: SessionStart with source="compact"

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
SOURCE=$(echo "$INPUT" | jq -r '.source // empty')

# Verify this is a post-compact restoration
if [ "$SOURCE" != "compact" ]; then
    echo '{"continue": true}'
    exit 0
fi

# Get repository context
REPO=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename 2>/dev/null || echo "default")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

# Skip if docugraph is not installed
if ! command -v docugraph &> /dev/null; then
    echo '{"continue": true}'
    exit 0
fi

# Initialize restore context
RESTORE_CONTEXT=""
ITEMS_RESTORED=0

# 1. Get user-stored context (from /remember, excluding internal _session_ keys)
USER_CONTEXT=$(docugraph memory show --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "{}")
if [ -n "$USER_CONTEXT" ] && [ "$USER_CONTEXT" != "{}" ] && [ "$USER_CONTEXT" != "No memories found." ]; then
    # Filter out internal session keys
    FILTERED_CONTEXT=$(echo "$USER_CONTEXT" | jq 'with_entries(select(.key | startswith("_session_") | not))' 2>/dev/null || echo "$USER_CONTEXT")
    if [ "$FILTERED_CONTEXT" != "{}" ] && [ "$FILTERED_CONTEXT" != "null" ]; then
        RESTORE_CONTEXT="${RESTORE_CONTEXT}### Stored Context (from /remember)\n\`\`\`json\n${FILTERED_CONTEXT}\n\`\`\`\n\n"
        ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
    fi
fi

# 2. Get most recent session files (from pre-compact saves)
LATEST_FILES_KEY=$(docugraph memory list --repo "$REPO" --branch "$BRANCH" --pattern "_session_files_%" 2>/dev/null | tail -1 || echo "")
if [ -n "$LATEST_FILES_KEY" ]; then
    RECENT_FILES=$(docugraph memory get "$LATEST_FILES_KEY" --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "")
    if [ -n "$RECENT_FILES" ] && [ "$RECENT_FILES" != "null" ]; then
        # Format files nicely
        FILES_LIST=$(echo "$RECENT_FILES" | jq -r '.[]' 2>/dev/null | head -20 || echo "$RECENT_FILES")
        if [ -n "$FILES_LIST" ]; then
            RESTORE_CONTEXT="${RESTORE_CONTEXT}### Files Modified Before Compression\n\`\`\`\n${FILES_LIST}\n\`\`\`\n\n"
            ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
        fi
    fi
fi

# 3. Get session summary
LATEST_SUMMARY_KEY=$(docugraph memory list --repo "$REPO" --branch "$BRANCH" --pattern "_session_summary_%" 2>/dev/null | tail -1 || echo "")
if [ -n "$LATEST_SUMMARY_KEY" ]; then
    SUMMARY=$(docugraph memory get "$LATEST_SUMMARY_KEY" --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "")
    if [ -n "$SUMMARY" ] && [ "$SUMMARY" != "null" ]; then
        RESTORE_CONTEXT="${RESTORE_CONTEXT}### Session Summary\n${SUMMARY}\n\n"
    fi
fi

# 4. Get any errors from before compression
LATEST_ERRORS_KEY=$(docugraph memory list --repo "$REPO" --branch "$BRANCH" --pattern "_session_errors_%" 2>/dev/null | tail -1 || echo "")
if [ -n "$LATEST_ERRORS_KEY" ]; then
    ERRORS=$(docugraph memory get "$LATEST_ERRORS_KEY" --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "")
    if [ -n "$ERRORS" ] && [ "$ERRORS" != "null" ] && [ "$ERRORS" != "[]" ]; then
        RESTORE_CONTEXT="${RESTORE_CONTEXT}### Errors Encountered Before Compression\n\`\`\`\n${ERRORS}\n\`\`\`\n\n"
        ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
    fi
fi

# 5. Get active architecture components
COMPONENTS=$(docugraph component list --repo "$REPO" --branch "$BRANCH" 2>/dev/null | head -15 || echo "")
if [ -n "$COMPONENTS" ] && [ "$COMPONENTS" != "No components found." ]; then
    RESTORE_CONTEXT="${RESTORE_CONTEXT}### Active Architecture Components\n${COMPONENTS}\n\n"
    ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
fi

# 6. Get active governance rules
RULES=$(docugraph rule list --repo "$REPO" --branch "$BRANCH" 2>/dev/null | head -10 || echo "")
if [ -n "$RULES" ] && [ "$RULES" != "No rules found." ]; then
    RESTORE_CONTEXT="${RESTORE_CONTEXT}### Active Governance Rules\n${RULES}\n\n"
    ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
fi

# 7. Get recent architectural decisions
DECISIONS=$(docugraph decision list --repo "$REPO" --branch "$BRANCH" 2>/dev/null | head -5 || echo "")
if [ -n "$DECISIONS" ] && [ "$DECISIONS" != "No decisions found." ]; then
    RESTORE_CONTEXT="${RESTORE_CONTEXT}### Recent Architectural Decisions\n${DECISIONS}\n\n"
    ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
fi

# 8. Get uncompleted todos from before compression
LATEST_TODOS_KEY=$(docugraph memory list --repo "$REPO" --branch "$BRANCH" --pattern "_session_todos_%" 2>/dev/null | tail -1 || echo "")
if [ -n "$LATEST_TODOS_KEY" ]; then
    TODOS=$(docugraph memory get "$LATEST_TODOS_KEY" --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "[]")
    if [ "$TODOS" != "[]" ] && [ "$TODOS" != "null" ] && [ -n "$TODOS" ]; then
        # Format todos nicely
        TODOS_FORMATTED=$(echo "$TODOS" | jq -r '.[] | "- [\(.status)] \(.content)"' 2>/dev/null || echo "$TODOS")
        if [ -n "$TODOS_FORMATTED" ]; then
            RESTORE_CONTEXT="${RESTORE_CONTEXT}### Uncompleted Tasks (from before compression)\n\`\`\`\n${TODOS_FORMATTED}\n\`\`\`\n\n**Action:** Use TodoWrite to restore these tasks if still relevant.\n\n"
            ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
        fi
    fi
fi

# 9. Get active plan from before compression
LATEST_PLAN_KEY=$(docugraph memory list --repo "$REPO" --branch "$BRANCH" --pattern "_session_plan_%" 2>/dev/null | tail -1 || echo "")
if [ -n "$LATEST_PLAN_KEY" ]; then
    PLAN_DATA=$(docugraph memory get "$LATEST_PLAN_KEY" --repo "$REPO" --branch "$BRANCH" 2>/dev/null || echo "{}")
    PLAN_PATH=$(echo "$PLAN_DATA" | jq -r '.path // empty' 2>/dev/null || echo "")
    PLAN_CONTENT=$(echo "$PLAN_DATA" | jq -r '.content // empty' 2>/dev/null || echo "")
    if [ -n "$PLAN_CONTENT" ]; then
        # Truncate plan if too long (keep first 2000 chars)
        PLAN_PREVIEW=$(echo "$PLAN_CONTENT" | head -c 2000)
        [ ${#PLAN_CONTENT} -gt 2000 ] && PLAN_PREVIEW="${PLAN_PREVIEW}...\n[truncated]"
        RESTORE_CONTEXT="${RESTORE_CONTEXT}### Active Plan (from before compression)\n**File:** \`${PLAN_PATH}\`\n\`\`\`markdown\n${PLAN_PREVIEW}\n\`\`\`\n\n**Action:** Review this plan and continue implementation if still relevant.\n\n"
        ITEMS_RESTORED=$((ITEMS_RESTORED + 1))
    fi
fi

# Output restored context
if [ -n "$RESTORE_CONTEXT" ] && [ "$ITEMS_RESTORED" -gt 0 ]; then
    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## Context Restored After Compression\n\nThe context window was just compressed. The following important context has been restored from DocuGraph (${ITEMS_RESTORED} categories):\n\n${RESTORE_CONTEXT}---\n\n**Tip:** Use \`/recall\` to see all stored context, or \`/remember key: value\` to store new context that survives compression."
  }
}
EOF
else
    cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "## Context Compression Complete\n\nThe context window was compressed. No previous context was found in DocuGraph to restore.\n\n**Tip:** Use \`/remember key: value\` to store important context that should survive future compressions."
  }
}
EOF
fi
