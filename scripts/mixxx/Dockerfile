# This is an evaluation runtime, not a publishable application image.
FROM ubuntu:26.04@sha256:065aa8626841f16057b78b027b464cd59c4fac36c204c72dcbf48d66f5fa8a96
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        mixxx=2.5.4+dfsg-1build1 python3 xvfb xdotool \
    && apt-get clean
ENV QT_QPA_PLATFORM=xcb QT_QUICK_BACKEND=software LANG=C.UTF-8
