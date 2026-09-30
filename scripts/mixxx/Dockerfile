# This is an evaluation runtime, not a publishable application image.
FROM ubuntu:26.04@sha256:da6fc2be547864451aa253836dd926da33623312df4a9a243e35dc877c378a78
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        mixxx=2.5.4+dfsg-1build1 python3 xvfb xdotool \
    && apt-get clean
ENV QT_QPA_PLATFORM=xcb QT_QUICK_BACKEND=software LANG=C.UTF-8
