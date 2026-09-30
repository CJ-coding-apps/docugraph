#!/bin/bash
# DocuGraph AI - Claude Code Integration Installer
#
# Usage: ./install.sh [target-directory]
#
# If no directory specified, installs to current directory.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Target directory (default: current directory)
TARGET_DIR="${1:-.}"

# Resolve to absolute path
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

echo -e "${GREEN}DocuGraph AI - Claude Code Integration Installer${NC}"
echo "================================================"
echo ""
echo "Source: $SCRIPT_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check for docugraph
if ! command -v docugraph &> /dev/null; then
    echo -e "${YELLOW}Warning: docugraph not found in PATH${NC}"
    echo "  Install with: pip install -e .   # from a checkout of this repository"
    echo ""
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not found in PATH${NC}"
    echo "  Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    echo ""
fi

# Create directories
echo "Creating directories..."
mkdir -p "$TARGET_DIR/.claude/skills"
mkdir -p "$TARGET_DIR/.claude/hooks"

# Copy skills
echo "Installing skills..."
for skill_dir in "$SCRIPT_DIR/skills"/*/; do
    skill_name=$(basename "$skill_dir")
    echo "  - $skill_name"
    mkdir -p "$TARGET_DIR/.claude/skills/$skill_name"
    cp "$skill_dir"* "$TARGET_DIR/.claude/skills/$skill_name/"
done

# Copy hooks
echo "Installing hooks..."
for hook_file in "$SCRIPT_DIR/hooks"/*.sh; do
    hook_name=$(basename "$hook_file")
    echo "  - $hook_name"
    cp "$hook_file" "$TARGET_DIR/.claude/hooks/"
    chmod +x "$TARGET_DIR/.claude/hooks/$hook_name"
done

# Handle settings.json
if [ -f "$TARGET_DIR/.claude/settings.json" ]; then
    echo ""
    echo -e "${YELLOW}Existing settings.json found!${NC}"
    echo "  Backing up to: .claude/settings.json.backup"
    cp "$TARGET_DIR/.claude/settings.json" "$TARGET_DIR/.claude/settings.json.backup"
    echo "  You may need to manually merge the hook configurations."
    echo ""
    echo "  New settings saved to: .claude/settings.json.docugraph"
    cp "$SCRIPT_DIR/settings.json" "$TARGET_DIR/.claude/settings.json.docugraph"
else
    echo "Installing settings.json..."
    cp "$SCRIPT_DIR/settings.json" "$TARGET_DIR/.claude/settings.json"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Installed to: $TARGET_DIR/.claude/"
echo ""
echo "Available skills:"
echo "  /index       - Index documentation"
echo "  /docs        - Search documentation"
echo "  /remember    - Store context"
echo "  /recall      - Retrieve context"
echo "  /architecture - Manage architecture"
echo "  /context     - Build rich context"
echo ""
echo "Active hooks:"
echo "  SessionStart   - Load stored context"
echo "  PreCompact     - Save context before compression"
echo "  Post-Compact   - Restore context after compression"
echo "  UserPromptSubmit - Auto-search docs"
echo "  PreToolUse     - Check architecture rules"
echo "  PostToolUse    - Auto-index doc changes"
echo "  Stop           - Prompt to save context"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code to load the new skills"
echo "  2. Index some documentation: /index https://your-docs.com"
echo "  3. Try searching: /docs your query"
echo ""
