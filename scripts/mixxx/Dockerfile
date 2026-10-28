# This is an evaluation runtime, not a publishable application image.
FROM ubuntu:26.04@sha256:0e73b71479e7bc5f006bca872e25c3cd680dc052a0116f90d63ae363537db0dd
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        mixxx=2.5.4+dfsg-1build1 python3 xvfb xdotool \
    && apt-get clean
ENV QT_QPA_PLATFORM=xcb QT_QUICK_BACKEND=software LANG=C.UTF-8
