FROM ubuntu:22.04

LABEL org.opencontainers.image.source="https://github.com/RBVI/ChimeraX"
LABEL org.opencontainers.image.description="UCSF ChimeraX headless runtime"

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies. This list mirrors the Ubuntu 22.04 entry in
# mkubuntu.py (the .deb package dependencies), plus libfftw3-single3,
# libopenjp2-7, and libxcb-xinerama0 which ChimeraX uses but mkubuntu.py
# does not currently list for 22.04.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libasound2 \
        libatk1.0-0 \
        libatomic1 \
        libbz2-1.0 \
        libc6 \
        libcairo-gobject2 \
        libcairo2 \
        libcrypt1 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libegl1 \
        libexpat1 \
        libffi8 \
        libfftw3-single3 \
        libfontconfig1 \
        libfreetype6 \
        libgbm1 \
        libgcc-s1 \
        libgdk-pixbuf-2.0-0 \
        libgfortran5 \
        libgl1 \
        libglib2.0-0 \
        libglu1-mesa \
        libgomp1 \
        libgssapi-krb5-2 \
        libgstreamer-gl1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer1.0-0 \
        libgtk-3-0 \
        liblzma5 \
        libncursesw6 \
        libnsl2 \
        libnspr4 \
        libnss3 \
        libopenjp2-7 \
        libosmesa6 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libpcsclite1 \
        libpulse0 \
        libspeechd2 \
        libsqlite3-0 \
        libssl3 \
        libstdc++6 \
        libtinfo6 \
        libtirpc3 \
        libuuid1 \
        libwayland-client0 \
        libwayland-cursor0 \
        libwayland-egl1 \
        libx11-6 \
        libx11-xcb1 \
        libxcb-cursor0 \
        libxcb-glx0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-render0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libxkbfile1 \
        libxrandr2 \
        libxrender1 \
        libxshmfence1 \
        libxtst6 \
        libzstd1 \
        xdg-utils \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY ChimeraX.app /opt/UCSF/ChimeraX.app

ENV PYOPENGL_PLATFORM=egl
ENV QT_QPA_PLATFORM=offscreen
ENV PATH="/opt/UCSF/ChimeraX.app/bin:${PATH}"

ENTRYPOINT ["ChimeraX", "--nogui", "--offscreen"]
