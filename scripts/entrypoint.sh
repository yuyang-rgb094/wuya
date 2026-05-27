#!/bin/bash
# =============================================================================
# WuYa Agents — Docker Entrypoint
# =============================================================================
set -e

# Default command
CMD="${1:-wuya}"
shift 2>/dev/null || true

# Load .env file if it exists
if [ -f /app/.env ]; then
    echo "[entrypoint] Loading configuration from /app/.env"
    export $(grep -v '^#' /app/.env | grep -v '^$' | xargs)
fi

# Display startup info
echo "============================================"
echo "  WuYa (无涯) Agents v$(python -c 'import wuya_agents; print(wuya_agents.__version__)')"
echo "  Environment: ${WUYA_ENVIRONMENT:-development}"
echo "  LLM Provider: ${WUYA_LLM_PROVIDER:-openai}"
echo "============================================"

# Execute the command
case "$CMD" in
    wuya)
        exec python -m wuya_agents.cli "$@"
        ;;
    serve)
        exec python -m wuya_agents.cli serve "$@"
        ;;
    evaluate)
        exec python -m wuya_agents.cli evaluate "$@"
        ;;
    batch)
        exec python -m wuya_agents.cli batch "$@"
        ;;
    test)
        echo "[entrypoint] Running tests..."
        exec pytest tests/ -v "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    *)
        exec "$CMD" "$@"
        ;;
esac
