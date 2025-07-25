# Base Jetson/Ubuntu
# FROM arm64v8/ubuntu:18.04
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3 este es mas para toch vision 


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VERSION=4.5.4

# ---------------------------------------------------
# Dependencias básicas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        pkg-config \
        python3.8 \
        python3.8-dev \
        python3-pip \
        cmake \
        git \
        curl \
        wget \
        unzip \
        gnupg \
        ca-certificates \
        libssl-dev \
        libcurl4-openssl-dev \
        libatlas-base-dev \
        liblapack-dev \
        libopenblas-dev \
        libboost-all-dev \
        libhdf5-dev \
        libcairo2-dev \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        v4l2loopback-utils \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libgtk2.0-dev && \
        # libopenmpi-dev \
        # libomp-dev \
        # ca-certificates \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------
# Establecer Python 3.8 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python3 -m pip install --upgrade pip

# ---------------------------------------------------
# Descargar y compilar OpenCV
# Copiar el Makefile para la instalación personalizada de OpenCV
COPY Makefile /app/opencv-install/Makefile

# Ejecutar la instalación con Makefile
WORKDIR /app/opencv-install
RUN make install_jetson


# ---------------------------------------------------
# Instalar Node.js 16
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------

# Crear directorio de trabajo
WORKDIR /app

# Copiar e instalar requirements
COPY requirements.txt /app/requirements.txt
# Instalar dependencias Python (excepto torch)
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore -r /app/requirements.txt

# ---------------------------------------------------
# Instalar TensorFlow 2.10
RUN python3 -m pip install --no-cache-dir tensorflow==2.10.0

# ---------------------------------------------------
# Descargar e instalar el paquete local de cuSparseLt para ARM64

# Instalar cuSparseLt para JetPack 4.6.1
RUN wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb && \
    dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb && \
    cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y libcusparselt0 libcusparselt-dev && \
    rm cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb

# Instalar PyTorch para JetPack 4.6.1 (JetPack 4.6.1 = JP 46.1 = v461)
ENV TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
RUN python3 -m pip install --no-cache-dir $TORCH_INSTALL
# ---------------------------------------------------
# Copiar proyecto
COPY . /app

# Descargar shape predictor
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && \
    mv shape_predictor_68_face_landmarks.dat /app/models/

# ---------------------------------------------------
# Crear usuario sin privilegios con acceso a /dev/videoX
RUN useradd -m appuser && \
    usermod -aG video appuser && \
    chown -R appuser:appuser /app

USER appuser

# Reinstalar requirements como appuser 
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore -r /app/requirements.txt

# Ejecutar la app
CMD ["python3", "main.py"]
