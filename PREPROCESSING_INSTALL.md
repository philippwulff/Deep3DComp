# Setting up the preprocessing pipeline

## 1. Install Dependencies:

Install [CLI11](https://github.com/CLIUtils/CLI11) following [this](https://github.com/facebookresearch/DeepSDF/issues/54#issuecomment-683285721):
```
git clone https://github.com/CLIUtils/CLI11
cd CLI11/
mkdir build
cd build
git submodule update --init
cmake ..
cmake --build .
sudo make install
```

Install [Eigen3](https://eigen.tuxfamily.org):
```
sudo apt install libeigen3-dev
```

Install [nanoflann](https://github.com/jlblancoc/nanoflann) (for Ubuntu 21.04 or newer: `sudo apt install libnanoflann-dev
`):
```
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
mkdir build
cd build
cmake ..
sudo make install
```

Install [Pangolin](https://github.com/stevenlovegrove/Pangolin) following [this](https://github.com/facebookresearch/DeepSDF/issues/81#issuecomment-953495747):
```
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
./scripts/install_prerequisites.sh recommended
git checkout v0.6
mkdir build && cd build
cmake ..
cmake --build .
sudo make install
```

## 2. Build the Preprocessing Binaries:

In the root directory of this repository run:
```
mkdir build
cd build
cmake .. -DCMAKE_CXX_STANDARD=17
make -j
```

# All Together

Or, alltogether (thanks to [Daniele Grattarola
](https://github.com/facebookresearch/DeepSDF/issues/88#issuecomment-1090011009)):

```
apt install git cmake build-essential libglfw3-dev libgles2-mesa-dev libgtest-dev libeigen3-dev

# Install CLI11
git clone https://github.com/CLIUtils/CLI11.git
cd CLI11
mkdir build
cd build
git submodule update --init
cmake ..
cmake --build .
sudo cmake --install .
cd ../..

# Install Pangolin
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
./scripts/install_prerequisites.sh all
git checkout v0.6
mkdir build && cd build
cmake ..
cmake --build .
sudo cmake --install .
cd ../..

# Install nanoflann
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
mkdir build && cd build
cmake ..
make
sudo make install
mkdir /usr/local/include/nanoflann
cp /usr/local/include/nanoflann.hpp /usr/local/include/nanoflann
cd ../..

# DeepSDF
git clone https://github.com/facebookresearch/DeepSDF.git
cd DeepSDF

###### Comment out line 97 of src/ShaderProgram.cpp
sed -i "97 s/^/\/\//" src/ShaderProgram.cpp

git submodule update --init
mkdir build && cd build
cmake .. -DCMAKE_CXX_STANDARD=17
make


# Run
export MESA_GL_VERSION_OVERRIDE=3.3
export PANGOLIN_WINDOW_URI=headless://
```
