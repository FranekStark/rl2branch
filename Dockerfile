FROM ubuntu:18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get -y upgrade && apt-get install -y \
python3.8 \
wget \
python3.8-distutils \
python3.8-dev \
build-essential \
git \
gfortran \
libtbb2 \
liblapack3

RUN apt-get remove -y cmake 
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.5/cmake-3.27.5-linux-x86_64.sh && chmod +x cmake-3.27.5-linux-x86_64.sh && /cmake-3.27.5-linux-x86_64.sh --prefix=/usr/local --skip-license

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-9 g++-9 cmake make && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90 && \
    apt-get clean

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py

WORKDIR /root
RUN wget https://www.scipopt.org/download/release/SCIPOptSuite-7.0.2-Linux-ubuntu.sh 
RUN chmod +x SCIPOptSuite-7.0.2-Linux-ubuntu.sh
RUN ./SCIPOptSuite-7.0.2-Linux-ubuntu.sh --skip-license --prefix=/root --include-subdir

ENV SCIPOPTDIR=/root/SCIPOptSuite-7.0.2-Linux

RUN python3.8 -m pip install pyscipopt==3.0.4

RUN python3.8 -m pip install cython wheel numpy scikit-build ninja

ENV CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:${SCIPOPTDIR}
RUN git clone https://github.com/FranekStark/ecole && cd ecole && mkdir wheels
WORKDIR /root/ecole
RUN python3.8 setup.py bdist_wheel --dist-dir wheels
RUN python3.8 -m pip install --no-index --find-links=wheels ecole

WORKDIR /root

ENV TORCH=1.9.0
ENV CUDA=cu102

RUN python3.8 -m pip install torch==${TORCH} torchvision==0.10.0 torchaudio --index-url https://download.pytorch.org/whl/${CUDA}

RUN python3.8 -m pip install torch-scatter==2.0.7 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python3.8 -m pip install torch-sparse==0.6.10 -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python3.8 -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python3.8 -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
RUN python3.8 -m pip install torch-geometric==1.7.2

RUN python3.8 -m pip install wandb

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SCIPOPTDIR}/lib


WORKDIR /root/rl2branch
ENV TYPE=mimpc

CMD []
