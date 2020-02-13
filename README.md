# Repository for Pytorch Docker image

Example for building and running the Docker (Using Docker 19.03)
```
docker build https://github.com/josueortc/pytorch-docker.git -t my/docker
docker run -it -p 8888:8888 --gpus '"device=0"' --mount type=bind,source=/home/josueortc/notebooks,destination=/notebooks/local_notebooks my/docker
```

This will create a jupyter lab
