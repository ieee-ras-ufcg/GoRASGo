# Get updates, if any, from the remote to local repo
git -C ~/GoRASGo pull > /dev/null 2>&1 || echo "[ERROR] Git Pull Failed"

# Start pigpiod daemon for advanced GPIO control
sudo pigpiod > /dev/null 2>&1 || echo "[ERROR] Pigpio Daemon Failed"

# Activate python virtual environment
source $HOME/.venv/bin/activate

# Run power management script, it stays running in the background
# even if the user logs out, avoiding hangups
sudo python $HOME/GoRASGo/Software/Python/gopigo3_power.py & > /dev/null 2>&1 || echo "[ERROR] GoPiGo3 Power Failed"

# Change for development directory for convenience
cd $HOME/GoRASGo/Software/Python/Tests/ > /dev/null 2>&1 || echo "[ERROR] Test Directory Not Found"