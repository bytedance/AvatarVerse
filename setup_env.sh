pip3 install -r requirements.txt

# parallel building core num for pytorch3d, faster building speed
export MAX_JOBS=8
pip3 install -v git+https://github.com/facebookresearch/pytorch3d.git
pip3 install -v git+https://github.com/NVlabs/nvdiffrast.git

sudo apt-get install -y libglew2.1 libglewmx-dev libglewmx1.1 freeglut3 libeigen3-dev
sudo apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libxi-dev openscad meshlab xvfb

