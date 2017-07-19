#!/usr/bin/env bash

if [ "$(whoami)" == "root" ]; then
  echo "running as root, please run as user you want to have stuff installed as"
  exit 1
fi

sudo apt-get update -y
sudo apt-get install -y git wget linux-image-generic build-essential unzip

sudo apt-get install  libglew-dev libcheese7 libcheese-gtk23 libclutter-gst-2.0-0 libcogl15 libclutter-gtk-1.0-0 libclutter-1.0-0  xserver-xorg-input-all


cd /tmp
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb

echo -e "\nexport CUDA_HOME=/usr/local/cuda\nexport CUDA_ROOT=/usr/local/cuda" >> ~/.bashrc
echo -e "\nexport PATH=/usr/local/cuda/bin:\$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc

sudo apt-get update
sudo apt-get install cuda-toolkit-7-0
sudo apt-get install nvidia-modprobe
pip install pycuda
