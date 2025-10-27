#!/bin/bash
# Helper script to update test snapshots after intentional changes
# Usage: ./update-snapshots.sh

set -e

echo "🔄 Starting server for snapshot updates..."
echo "   (Make sure no other instance is running on port 7860)"

# Start server in background
python ../../../webui.py --deforum-api --skip-prepare-environment &
SERVER_PID=$!

# Ensure server is killed on exit
trap "echo '🛑 Stopping server...'; kill $SERVER_PID 2>/dev/null || true" EXIT

# Wait for server to be ready
echo "⏳ Waiting for server to start..."
sleep 10

# Update snapshots
echo "📸 Updating snapshots..."
pytest tests/integration/api_test.py tests/integration/postprocess_test.py --snapshot-update -v

echo ""
echo "✅ Snapshots updated!"
echo ""
echo "📋 Next steps:"
echo "   1. Review changes: git diff tests/__snapshots__/"
echo "   2. If correct, commit: git add tests/__snapshots__/ && git commit -m 'test: Update snapshots for [reason]'"
echo "   3. If incorrect, revert: git checkout tests/__snapshots__/"
