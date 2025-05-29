# Get updates, if any, from the remote to local repo
git -C ~/GoRASGo pull > /dev/null 2>&1 || echo -e "\n[ERROR] Git Pull Failed"

# Start pigpiod daemon for advanced GPIO control
sudo pigpiod > /dev/null 2>&1 || echo "[ERROR] Pigpio Daemon Failed"

# Activate python virtual environment
source $HOME/.venv/bin/activate > /dev/null 2>&1 || echo "[ERROR] Venv Activation Failed"

# Run power management script in background only once per boot
if [ ! -f /tmp/gopigo3_power.lock ]; then
    nohup sudo python "$HOME/GoRASGo/Software/Python/gopigo3_power.py" > /dev/null 2>&1 &
    touch /tmp/gopigo3_power.lock
fi

# Change for development directory for convenience
cd $HOME/GoRASGo/Software/Python/Tests/ > /dev/null 2>&1 || echo "[ERROR] Test Directory Not Found"