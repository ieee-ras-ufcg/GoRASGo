# Start pigpiod daemon for advanced GPIO control
sudo pigpiod

# Activate python virtual environment
source $HOME/.venv/bin/activate

# Run power management script, it stays running in the background
# even if the user logs out, avoiding hangups
python $HOME/GoRASGo/Software/Python/gopigo3_power &

# Change for development directory for convenience
cd $HOME/GoRASGo/Software/Python/Tests/