#!/bin/bash

# Update and upgrade the system
sudo apt update
# sudo apt upgrade -y

# Install Python 3.8 and necessary Python tools
# sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip git s3fs
sudo apt install -y python3-pip git s3fs screen
python -m pip install --upgrade pip

# Add Python 3.8 to the alternatives system and set it as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --set python3 /usr/bin/python3.8

# Create a symbolic link for python to point to python3 and pip3
sudo ln -s /usr/bin/python3.8 /usr/bin/python
sudo ln -s /usr/bin/pip3 /usr/bin/pip
# Verify pip installation
pip3 --version

# Clean up
sudo apt autoremove -y
sudo apt clean


# Clone the SCALAE repository
git clone https://github.com/bazzile/SCALAE.git && cd SCALAE && git checkout dev

# Install the required Python packages
pip install -r requirements.txt
