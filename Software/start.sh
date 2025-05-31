#! /bin/bash

# Checkout to test branch
git -C $HOME/GoRASGo checkout test > /dev/null 2>&1 || echo "[ERROR] Checkout Failed"

# Get updates, if any, from the remote to local repo
git -C $HOME/GoRASGo pull > /dev/null 2>&1 || echo -e "\n[ERROR] Git Pull Failed"

# Start pigpiod daemon for advanced GPIO control
sudo pigpiod > /dev/null 2>&1 || echo "[ERROR] Pigpio Daemon Failed"

# Activate python virtual environment
source $HOME/.venv/bin/activate > /dev/null 2>&1 || echo "[ERROR] Venv Activation Failed"

# Change for development directory for convenience
cd $HOME/GoRASGo/Software/Python/Tests/ > /dev/null 2>&1 || echo "[ERROR] Test Directory Not Found"