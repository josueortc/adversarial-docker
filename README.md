# Repository for Pytorch Docker image

Example for building and running the Docker (Using Docker 19.03) with nvidia-docker2
```
docker build https://github.com/josueortc/adversarial-docker.git -t adversarial
docker run -it -p 8888:8888 --gpus '"device=0"' --mount type=bind,source=/mnt/savefiles/sda/josueortc/adversarial_notebooks,destination=/notebooks/local_notebooks adversarial
```

This will create a jupyter lab
