# Base Jetson/Ubuntu
FROM arm64v8/ubuntu:18.04

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
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------
# Establecer Python 3.8 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python3 -m pip install --upgrade pip

# ---------------------------------------------------
# Descargar y compilar OpenCV
WORKDIR /opt
RUN git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git

WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D PYTHON_EXECUTABLE=$(which python3) \
    -D BUILD_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Eliminar fuentes de OpenCV tras la instalación
RUN rm -rf /opt/opencv /opt/opencv_contrib

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

# Instalar PyTorch + torchvision Jetson (JetPack 4.6)
RUN python3 -m pip install --no-cache-dir \
    https://nvidia.box.com/shared/static/p57jwntvnv79zhw6cptl3p1z0ocl9g3i.whl \
    https://nvidia.box.com/shared/static/8t2d7b6e4p7x1r1ypi09vxx2blf5sfyd.whl

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

# Reinstalar requirements como appuser (opcional)
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore -r /app/requirements.txt

# Ejecutar la app
CMD ["python3", "main.py"]
