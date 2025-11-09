# Prerequisites

This guide covers everything you need to install on your Mac before setting up the Interview Analyzer project.

## Required Tools

- macOS 12.0 (Monterey) or later
- ~20 GB free disk space
- Admin access to your Mac

---

## 1. Install Homebrew

Homebrew is macOS's package manager. We'll use it to install Git and other tools.

### Installation

1. Open **Terminal** (Applications → Utilities → Terminal)

2. Run this command:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. Follow the on-screen instructions:
   - You'll be prompted to press **Enter/Return** to continue
   - Enter your **Mac password** when prompted (you won't see characters as you type - this is normal)
   - Installation downloads **~400 MB** and takes **2-5 minutes**
   - You'll see: `Installation successful!` when complete

4. After installation, you'll see instructions to add Homebrew to your PATH:
   
   **On Apple Silicon Macs (M1/M2/M3)**, run these two commands exactly as shown:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

   **On Intel Macs**, Homebrew is automatically added to your PATH (no action needed).

### Verification

```bash
brew --version
```

**Expected output:** Something like `Homebrew 4.x.x`

---

## 2. Install Git

Git is required for cloning the repository.

### Installation

```bash
brew install git
```

### Configure Git (First Time Only)

Set your name and email (used for commits). Replace with your actual information:

```bash
git config --global user.name "Jane Doe"
git config --global user.email "jane.doe@example.com"
```

**Example for a specific user:**
```bash
git config --global user.name "Alex Rivera"
git config --global user.email "alex.rivera@company.com"
```

### Verification

```bash
git --version
```

**Expected output:** `git version 2.x.x` or higher

---

## 3. Install Docker Desktop

Docker Desktop provides the containerization platform for running all project services.

### Installation

1. **Download Docker Desktop:**
   - Visit: https://www.docker.com/products/docker-desktop
   - Click "Download for Mac"
   - Choose the correct version:
     - **Apple Silicon** (M1/M2/M3): Download "Mac with Apple chip"
     - **Intel**: Download "Mac with Intel chip"

2. **Install:**
   - Open the downloaded `.dmg` file
   - Drag Docker icon to Applications folder
   - Open Docker from Applications
   - **First launch takes 2-3 minutes** to initialize
   
   **You'll be prompted for permissions:**
   - **"Docker wants to install helper tools"** - Click "OK" and enter your Mac password
   - **"Allow Docker to access files"** - This allows Docker to read/write project files
   - **"Docker wants to make changes"** - Required for container networking

3. **Configure Docker Desktop:**
   
   Open Docker Desktop → Settings (gear icon) → Resources:
   
   - **CPUs:** At least 4 (recommended: 6)
   - **Memory:** At least 8 GB (recommended: 12 GB)
   - **Disk:** At least 60 GB (project will use ~10-15 GB)
   - **Swap:** 1 GB
   
   Click "Apply & Restart" (takes 30-60 seconds)

4. **Enable Docker Compose V2:**
   
   Go to Settings → General:
   - ✅ Check "Use Docker Compose V2"
   
   **Note:** Docker Compose V2 is required (v2.0.0 or later).

### Verification

1. Check Docker is running (whale icon in menu bar should be steady, not animated)

2. Run these commands in Terminal:
   ```bash
   docker --version
   docker compose version
   ```

   **Expected output:**
   ```
   Docker version 24.x.x or higher
   Docker Compose version v2.x.x or higher
   ```

3. Test Docker functionality:
   ```bash
   docker run hello-world
   ```

   **Expected output:** Should see "Hello from Docker!" message

---

## 4. Install Cursor IDE

Cursor is an AI-powered code editor based on VS Code. This project is optimized for Cursor.

### Installation

1. **Download Cursor:**
   - Visit: https://cursor.sh
   - Click "Download for Mac"
   - Choose your Mac type (Apple Silicon or Intel)

2. **Install:**
   - Open the downloaded `.dmg` file
   - Drag Cursor to Applications folder
   - Open Cursor from Applications
   - If you see a security warning: System Settings → Privacy & Security → Click "Open Anyway"

3. **First Launch Setup:**
   - Sign in or create a Cursor account (optional but recommended)
   - Choose your theme preference
   - Import settings from VS Code if you have them (optional)

### Recommended Extensions

Install these extensions in Cursor for the best development experience:

1. **Python Extension** (Microsoft)
   - Open Command Palette: `Cmd+Shift+P`
   - Type: "Extensions: Install Extensions"
   - Search for "Python" by Microsoft (Extension ID: `ms-python.python`)
   - Click "Install"

2. **Docker Extension** (Microsoft)
   - Search for "Docker" by Microsoft (Extension ID: `ms-azuretools.vscode-docker`)
   - Click "Install"

3. **YAML Extension** (Red Hat)
   - Search for "YAML" by Red Hat (Extension ID: `redhat.vscode-yaml`)
   - Click "Install"

4. **Pylance** (Microsoft)
   - Search for "Pylance" (Extension ID: `ms-python.vscode-pylance`)
   - Click "Install"

**Note:** These extensions typically take 10-30 seconds each to install.

### Verification

1. Open Cursor
2. Open Command Palette: `Cmd+Shift+P`
3. Type: "Python: Select Interpreter"
4. You should see a list of Python interpreters (even if none are selected yet)

---

## 5. Get API Keys

You'll need API keys for the AI services used in this project.

### OpenAI API Key

1. **Create Account:**
   - Visit: https://platform.openai.com/signup
   - Sign up with email or Google/Microsoft account

2. **Add Payment Method:**
   - Go to: https://platform.openai.com/account/billing/overview
   - Add a payment method (credit card required)
   - **Recommended:** Set a usage limit of $20/month under "Usage limits" to avoid unexpected charges
   - Note: Development typically costs $2-5/month

3. **Create API Key:**
   - Go to: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Name it: "Interview Analyzer Dev"
   - Copy the key (starts with `sk-proj-` or `sk-`)
   - **SAVE IT IMMEDIATELY** - you won't be able to see it again
   - **Store in password manager:** Use 1Password, LastPass, Bitwarden, or macOS Keychain
   - Or create a secure note in your password manager labeled "Interview Analyzer API Keys"

### Gemini API Key

1. **Access Google AI Studio:**
   - Visit: https://makersuite.google.com/app/apikey
   - Sign in with your Google account

2. **Create API Key:**
   - Click "Create API key"
   - Select "Create API key in new project" (or choose an existing project)
   - Copy the key (starts with `AIzaSy`)
   - **SAVE IT IMMEDIATELY**
   - **Store in same password manager entry** as your OpenAI key

### Store Keys for Next Step

In your password manager (1Password, LastPass, etc.), create an entry labeled:
**"Interview Analyzer API Keys"**

With this format:
```
OPENAI_API_KEY=[paste-your-actual-key-here]
GEMINI_API_KEY=[paste-your-actual-key-here]
```

**Alternative:** Create a secure note in macOS Notes:
1. Open Notes app
2. File → New Note
3. Paste the keys above
4. File → Lock Note (enter password)

You'll copy these values into your `.env` file in the next section.

---

## Verification Checklist

Before proceeding, verify you have:

- [ ] Homebrew installed and in PATH
- [ ] Git installed and configured with your name/email
- [ ] Docker Desktop installed, running, and configured (8GB+ RAM, 60GB+ disk)
- [ ] Docker and Docker Compose working (`docker run hello-world` succeeds)
- [ ] Cursor IDE installed and opened
- [ ] Python, Docker, YAML, and Pylance extensions installed in Cursor
- [ ] OpenAI API key obtained and saved
- [ ] Gemini API key obtained and saved

## Troubleshooting

### "Command not found: brew"
- Make sure you ran the PATH setup command after installing Homebrew
- Close and reopen Terminal
- On Apple Silicon: Check that `/opt/homebrew/bin` is in your PATH

### Docker Desktop won't start
- Check System Requirements: macOS 11 or later
- Restart your Mac
- Reinstall Docker Desktop
- Check for macOS updates

### Cursor security warning
- System Settings → Privacy & Security → scroll down → Click "Open Anyway"
- Or: Right-click Cursor in Applications → Open → Click "Open"

## What's Next?

You've completed the prerequisites! 

Next: [Initial Setup →](./02-initial-setup.md)

Learn how to clone the repository and configure your environment.

