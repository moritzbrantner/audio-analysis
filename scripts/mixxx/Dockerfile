# This is an evaluation runtime, not a publishable application image.
FROM ubuntu:26.04@sha256:d66b366dac61b575bee2aef07e9c26e8070096fbe84a3d4d3eb0da069b5d76db
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        mixxx=2.5.4+dfsg-1build1 python3 xvfb xdotool \
    && apt-get clean
ENV QT_QPA_PLATFORM=xcb QT_QUICK_BACKEND=software LANG=C.UTF-8
