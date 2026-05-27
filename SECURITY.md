# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in WuYa Agents, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Send an email to: **security@wuya.dev** (or the project maintainer's email)
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- We will acknowledge receipt within 48 hours
- We will provide an initial assessment within 72 hours
- We will keep you updated on the fix progress
- We will credit you in the security advisory (unless you prefer anonymity)

### Responsible Disclosure

We follow a coordinated disclosure process:

1. **Report received** → Acknowledge and triage
2. **Assessment** → Validate and determine severity
3. **Fix development** → Develop and test the fix
4. **Release** → Publish the fix and security advisory
5. **Credit** → Acknowledge the reporter

## Security Best Practices

When using WuYa Agents:

- **Never commit API keys** to version control. Use `.env` files (already in `.gitignore`)
- **Keep dependencies updated**: `pip install --upgrade wuya-agents`
- **Review dependencies**: `pip audit` or `safety check`
- **Use environment isolation**: virtual environments or Docker containers
- **Limit API permissions**: Use API keys with minimal required scopes

## API Key Security

WuYa Agents requires API keys for LLM providers. Protect these keys:

```bash
# ✅ Good: Use environment variables
export OPENAI_API_KEY="sk-..."

# ✅ Good: Use .env file (not committed to git)
echo "OPENAI_API_KEY=sk-..." >> .env

# ❌ Bad: Hardcode in source code
api_key = "sk-..."
```

## Dependency Security

We regularly audit dependencies for known vulnerabilities. To check your installation:

```bash
pip install pip-audit
pip-audit
```
