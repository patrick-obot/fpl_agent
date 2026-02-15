#!/bin/bash
# One-time setup for FPL Review downloader on Raspberry Pi
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "FPL Pi Downloader Setup"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p data data/browser_profile logs

# Create Python venv
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "venv already exists, skipping."
fi

# Install dependencies
echo "Installing Python dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# Install Playwright Chromium
echo "Installing Playwright Chromium browser..."
venv/bin/playwright install chromium
venv/bin/playwright install-deps chromium 2>/dev/null || true

# .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo ""
    echo ">>> EDIT .env with your credentials: nano $SCRIPT_DIR/.env"
else
    echo ".env already exists, skipping."
fi

# SSH key check
SSH_KEY="$HOME/.ssh/fpl_pi_to_vps"
if [ -f "$SSH_KEY" ]; then
    echo "SSH key found at $SSH_KEY"
else
    echo ""
    echo ">>> No SSH key at $SSH_KEY"
    echo "    Generate one with: ssh-keygen -t ed25519 -f $SSH_KEY -N \"\""
    echo "    Copy to VPS with:  ssh-copy-id -i ${SSH_KEY}.pub root@YOUR_VPS_IP"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env:          nano $SCRIPT_DIR/.env"
echo "  2. First run (interactive, for Patreon login):"
echo "     $SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/download_and_upload.py --no-headless"
echo "  3. After login works, add cron jobs:"
echo "     crontab -e"
echo "     # Daily at 06:00 UTC"
echo "     0 6 * * * $SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/download_and_upload.py >> $SCRIPT_DIR/logs/cron.log 2>&1"
echo "     # Retry at 08:00 UTC"
echo "     0 8 * * * $SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/download_and_upload.py >> $SCRIPT_DIR/logs/cron.log 2>&1"
echo "     # Extra run midnight UTC on Fri/Sat (deadline days)"
echo "     0 0 * * 5,6 $SCRIPT_DIR/venv/bin/python $SCRIPT_DIR/download_and_upload.py >> $SCRIPT_DIR/logs/cron.log 2>&1"
