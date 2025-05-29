# Start pigpiod daemon for advanced GPIO control
sudo pigpiod

# Activate python virtual environment
source ~/.venv/bin/activate

# Run power management script, it stays running in the background
# even if the user logs out, avoiding hangups
nohup python -m ~/GoRASGo/Software/Python/gopigo3_power &

# Change for development directory for convenience
cd ~/GoRASGo/Software/Python/Tests/