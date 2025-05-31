#!/bin/bash

echo "Enabling all interfaces..."
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_serial 0
sudo raspi-config nonint do_onewire 0
sudo raspi-config nonint do_ssh 0
sudo raspi-config nonint do_vnc 0

echo "Installing Python requirements..."
cd $HOME/GoRASGo/Software/Python
python -m venv $HOME/.venv
source $HOME/.venv/bin/activate
pip install --upgrade pip setuptools wheel > /dev/null
pip install -r requirements.txt || { echo "[ERROR] Failed to install requirements"; exit 1; }

echo "Installing GoRASGo Python package..."
pip install . || { echo "[ERROR] Failed to install GoRASGo package"; exit 1; }

echo "Adding start script to .bashrc if not already present..."
START_LINE="source $HOME/GoRASGo/Software/start.sh"
if ! grep -Fxq "$START_LINE" $HOME/.bashrc; then
    echo "$START_LINE" >> $HOME/.bashrc
    echo "Added start.sh to .bashrc"
else
    echo "start.sh already configured in .bashrc"
fi

echo "Rebooting in 5 seconds..."
sleep 5
sudo reboot