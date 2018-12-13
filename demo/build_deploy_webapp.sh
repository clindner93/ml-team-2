#!/usr/bin/env bash

# read common variables (between installation, training, and serving)
source variables.bash



# docker authorization
if [ "${DOCKER_HUB}" = "docker.io" ]
then
    sudo docker login
fi

# move to webapp folder
cd webapp

# build an image passing correct IP and port
sudo docker build . --no-cache  -f Dockerfile -t ${CLIENT_IMAGE}
sudo docker push ${CLIENT_IMAGE}
cd ..

./deploy_webapp.sh