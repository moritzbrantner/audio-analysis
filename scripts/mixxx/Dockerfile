# This is an evaluation runtime, not a publishable application image.
FROM ubuntu:26.04@sha256:c098726dd9b9733f3e885f8de0ef375206a311b0a317eadc40d347b1115fb944
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        mixxx=2.5.4+dfsg-1build1 python3 xvfb xdotool \
    && apt-get clean
ENV QT_QPA_PLATFORM=xcb QT_QUICK_BACKEND=software LANG=C.UTF-8
