# Repository for Pytorch Docker image

Example for building and running the Docker (Using Docker 19.03)
```
docker build https://github.com/josueortc/pytorch-docker.git -t my/docker
docker run -p 8888:8888 --gpus 0 -t my/docker
```

This will create a jupyter lab
