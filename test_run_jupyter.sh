#!/usr/bin/env bash

# build base image
docker build -t corex_tag .

docker build -t corex_tag_interactive -f Dockerfile-jupyter .
docker run -p 8990:8888 \
    -v "$(pwd)/.":/opt/albergo/.\
    corex_tag_interactive