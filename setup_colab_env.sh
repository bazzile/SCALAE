#install python 3.9 and dev utils
#you may not need all the dev libraries, but I haven't tested which aren't necessary.
sudo apt-get update -y
sudo apt-get install python3.8 python3.8-dev python3.8-distutils libpython3.8-dev

#change alternatives
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

#Check that it points at the right location
python3 --version

# install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --force-reinstall

#install colab's dependencies
python3 -m pip install ipython ipython_genutils ipykernel jupyter_console prompt_toolkit httplib2 astor

# link to the old google package
ln -s /usr/local/lib/python3.10/dist-packages/google \
       /usr/local/lib/python3.8/dist-packages/google

# There has got to be a better way to do this...but there's a bad import in some of the colab files
# IPython no longer exposes traitlets like this, it's a separate package now
sed -i "s/from IPython.utils import traitlets as _traitlets/import traitlets as _traitlets/" /usr/local/lib/python3.8/dist-packages/google/colab/*.py
sed -i "s/from IPython.utils import traitlets/import traitlets/" /usr/local/lib/python3.8/dist-packages/google/colab/*.py


# !git clone https://github.com/bazzile/SCALAE.git && cd SCALAE && git checkout dev
# %cd SCALAE
# !chmod +x setup_colab_env.sh && ./setup_colab_env.sh && pip install -r requirements.txt
# !mkdir /data && ln -s /content/drive/MyDrive/scalae_project/data/* /data