#!/bin/bash

echo "Enabling all interfaces..."
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_serial 0
sudo raspi-config nonint do_onewire 0
sudo raspi-config nonint do_ssh 0
sudo raspi-config nonint do_vnc 0

echo "[INFO] Setting Python development environment..."
cd $HOME/GoRASGo/Software/Python
echo "[INFO] Creating Python Virtual Environment..."
python -m venv $HOME/.venv
echo "[INFO] Activating Python Virtual Environment..."
source $HOME/.venv/bin/activate
echo "[INFO] Installing Python requirements..."
pip install --upgrade pip setuptools wheel > /dev/null
pip install -r requirements.txt || { echo "[ERROR] Failed to install requirements"; exit 1; }
echo "[INFO] Installing GoRASGo Python package..."
pip install . || { echo "[ERROR] Failed to install GoRASGo package"; exit 1; }

echo "[INFO] Adding start script to .bashrc if not already present..."
START_LINE="source $HOME/GoRASGo/Software/start.sh"
if ! grep -Fxq "$START_LINE" $HOME/.bashrc; then
    echo "$START_LINE" >> $HOME/.bashrc
    echo "[INFO] Added start.sh to .bashrc"
else
    echo "[INFO] start.sh already configured in .bashrc"
fi

echo "[INFO] Installation Finished"
echo "[INFO] Reboot for changes to take effect"