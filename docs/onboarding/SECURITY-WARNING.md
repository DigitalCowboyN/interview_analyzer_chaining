# ⚠️ CRITICAL SECURITY WARNING ⚠️

## Exposed API Keys Detected

**Date Identified:** November 9, 2025

### Issue

Real API keys have been committed to this repository in the following files:

1. `.env` - Contains actual OpenAI and Gemini API keys
2. `.devcontainer/devcontainer.env` - Contains duplicate keys

**These files should NEVER contain real secrets in a Git repository.**

### Immediate Actions Required

#### For Repository Owner/Maintainer:

1. **Rotate ALL exposed API keys immediately:**

   **OpenAI:**

   - Go to: https://platform.openai.com/api-keys
   - Find your exposed key (check Git history or security scan results)
   - Click "Revoke" or delete the key
   - Create a new key
   - Update your local `.env` file only (do NOT commit)

   **Gemini:**

   - Go to: https://makersuite.google.com/app/apikey
   - Find your exposed key (check Git history or security scan results)
   - Delete the key
   - Create a new key
   - Update your local `.env` file only (do NOT commit)

2. **Check for unauthorized usage:**

   - Review OpenAI usage: https://platform.openai.com/usage
   - Review Gemini usage in Google Cloud Console
   - Look for unexpected API calls or usage spikes
   - If compromised, report to the platform

3. **Remove keys from Git history:**

   **Option A - Using BFG Repo-Cleaner (Recommended):**

   ```bash
   # Install BFG
   brew install bfg

   # Clone a fresh copy
   git clone --mirror https://github.com/DigitalCowboyN/interview_analyzer_chaining.git

   # Remove sensitive data
   cd interview_analyzer_chaining.git
   bfg --replace-text secrets.txt  # Create secrets.txt with keys to remove

   # Clean up
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive

   # Force push (requires force push permissions)
   git push --force
   ```

   **Option B - Using git-filter-repo:**

   ```bash
   # Install git-filter-repo
   brew install git-filter-repo

   # Clone repo
   git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining.git
   cd interview_analyzer_chaining

   # Remove sensitive files from history
   git filter-repo --invert-paths --path .env --path .devcontainer/devcontainer.env

   # Force push
   git push origin --force --all
   ```

4. **Update repository to use template:**

   ```bash
   # Remove .env from tracking (but keep local copy)
   git rm --cached .env
   git rm --cached .devcontainer/devcontainer.env

   # Ensure .gitignore is correct
   echo ".env" >> .gitignore
   echo ".devcontainer/devcontainer.env" >> .gitignore

   # Commit the changes
   git add .gitignore .env.example
   git commit -m "Security: Remove real API keys, add .env.example template"
   git push
   ```

#### For New Developers:

**DO NOT use the keys from `.env` or `.devcontainer/devcontainer.env` if they exist in the repository.**

Instead:

1. Create your own API keys:

   - OpenAI: https://platform.openai.com/api-keys
   - Gemini: https://makersuite.google.com/app/apikey

2. Copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your personal keys:

   ```bash
   cursor .env  # or: nano .env
   ```

4. **NEVER commit your `.env` file**

### Why This Matters

**Exposed API keys can lead to:**

- ✗ Unauthorized API usage charged to your account
- ✗ Quota exhaustion (your app stops working)
- ✗ Data breaches (if keys have broad permissions)
- ✗ Service suspension by the API provider
- ✗ Financial loss (API usage can be expensive)

**Real-world impact:**

- OpenAI charges per token (~$0.15-$2 per 1M tokens for GPT-4)
- Malicious actors can rack up thousands in charges
- Keys are scraped by bots within minutes of exposure

### Prevention Going Forward

1. **Never commit `.env` files**

   - Always use `.env.example` as template
   - Add `.env` to `.gitignore`
   - Double-check before every commit: `git status`

2. **Use pre-commit hooks:**

   ```bash
   # Install pre-commit
   brew install pre-commit

   # Add to .pre-commit-config.yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
         - id: detect-private-key
         - id: check-added-large-files

   # Install hooks
   pre-commit install
   ```

3. **Use environment-specific practices:**

   - **Local development:** Store keys in `.env` (gitignored)
   - **CI/CD:** Use GitHub Secrets, GitLab CI Variables
   - **Production:** Use AWS Secrets Manager, Azure Key Vault, etc.

4. **Regular audits:**

   ```bash
   # Scan for secrets in repository
   git log -p | grep -E "\[YOUR_" | grep -v "example"  # Look for [YOUR_ pattern
   git log -p | grep -iE "api.?key|password|secret" | grep -v "example"  # Look for common patterns

   # Use tools like gitleaks or truffleHog
   brew install gitleaks
   gitleaks detect --source .
   ```

### How .env.example Works

The `.env.example` file is a **template** that:

- ✓ Contains all required environment variable names
- ✓ Includes documentation for each variable
- ✓ Uses placeholder values (no real secrets)
- ✓ Can be safely committed to Git
- ✓ Guides developers on what values to provide

**Workflow:**

```bash
# Developer clones repo
git clone https://github.com/DigitalCowboyN/interview_analyzer_chaining.git

# Copy template
cp .env.example .env

# Edit with real values
cursor .env  # Add your personal API keys

# .env is gitignored, so git won't track it
git status  # Should NOT show .env
```

### Resources

- **OWASP Secrets Management:** https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html
- **GitHub Secret Scanning:** https://docs.github.com/en/code-security/secret-scanning
- **Git-filter-repo:** https://github.com/newren/git-filter-repo
- **BFG Repo-Cleaner:** https://rtyley.github.io/bfg-repo-cleaner/

### Questions?

If you've discovered exposed keys or need help with remediation, contact your team lead immediately.

**Do NOT discuss specific key values in public channels (Slack, email, etc.).**

---

**This warning will remain until the exposed keys are rotated and removed from Git history.**
