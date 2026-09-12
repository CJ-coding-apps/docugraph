#!/bin/bash
# DocuGraph Auto-Index Hook
# Automatically re-indexes documentation files when they are modified
#
# This hook runs at PostToolUse for Edit|Write operations and queues
# documentation files for re-indexing in the background.

set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)

# Extract file path
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Skip if we don't have a file path
if [ -z "$FILE_PATH" ]; then
    echo '{"continue": true}'
    exit 0
fi

# Skip if docugraph is not installed
if ! command -v docugraph &> /dev/null; then
    echo '{"continue": true}'
    exit 0
fi

# Check if this is a documentation file
IS_DOC_FILE=false

# Check by extension
if echo "$FILE_PATH" | grep -qiE '\.(md|mdx|rst|txt|adoc|asciidoc)$'; then
    IS_DOC_FILE=true
fi

# Check by path (common documentation directories)
if echo "$FILE_PATH" | grep -qiE '(^docs?/|/docs?/|^documentation/|/documentation/|README|CHANGELOG|CONTRIBUTING|LICENSE)'; then
    IS_DOC_FILE=true
fi

# Check for API documentation
if echo "$FILE_PATH" | grep -qiE '(openapi|swagger|api-spec)\.(ya?ml|json)$'; then
    IS_DOC_FILE=true
fi

# If not a documentation file, skip
if [ "$IS_DOC_FILE" = false ]; then
    echo '{"continue": true}'
    exit 0
fi

# Get the file's directory for indexing
FILE_DIR=$(dirname "$FILE_PATH")

# Queue for background indexing (non-blocking)
# We index just the single file to be efficient
(
    # Small delay to ensure file is fully written
    sleep 1

    # Index the specific file
    docugraph index local "$FILE_PATH" --quiet 2>/dev/null || true

    # Log the indexing (optional, for debugging)
    # echo "$(date): Indexed $FILE_PATH" >> ~/.docugraph/auto-index.log
) &

# Respond immediately (don't wait for indexing)
cat << EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "additionalContext": "Documentation file detected. Queued for automatic re-indexing: \`${FILE_PATH}\`"
  }
}
EOF
