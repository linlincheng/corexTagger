#!/usr/bin/env bash

docker build -t corex_tag .

if [ ! "$(docker ps -q -f name=corex_tag)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=corex_tag)" ]; then
        # cleanup
        docker rm corex_tag
    fi
    # run your container
    docker run -d -P --name corex_tag -v "$(pwd)/.":/opt/corex_tag/ \
  -v "$(pwd)/.":/opt/corex_tag/ \
  -v "$(pwd)/.":/opt/corex_tag/ corex_tag
fi
