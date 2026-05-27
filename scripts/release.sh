#!/bin/bash
# =============================================================================
# WuYa Agents — Release Script
# =============================================================================
# Usage: ./scripts/release.sh [version]
# Example: ./scripts/release.sh 0.1.0
# =============================================================================

set -e

VERSION=${1:-0.1.0}
echo "============================================"
echo "  WuYa Agents Release Script v${VERSION}"
echo "============================================"

# Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

if ! command -v python &> /dev/null; then
    echo "❌ Python is required"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git is required"
    exit 1
fi

# Clean previous builds
echo ""
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install build tools
echo ""
echo "📦 Installing build tools..."
pip install --quiet build twine

# Build package
echo ""
echo "🔨 Building package..."
python -m build

# Check package
echo ""
echo "✅ Checking package..."
twine check dist/*

# Show package info
echo ""
echo "📊 Package info:"
ls -lh dist/

echo ""
echo "============================================"
echo "  Build successful!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Test locally: pip install dist/wuya_agents-${VERSION}-py3-none-any.whl"
echo "  2. Upload to PyPI: twine upload dist/*"
echo "  3. Or push to GitHub and let GitHub Actions handle it"
echo ""
echo "To upload manually:"
echo "  twine upload dist/*"
echo ""
