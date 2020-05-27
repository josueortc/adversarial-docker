# Repository for Pytorch Docker image

Example for building and running the Docker (Using Docker 19.03) with nvidia-docker2
```
git clone https://github.com/josueortc/adversarial-docker
cd adversarial-docker
docker build https://github.com/josueortc/pytorch-docker.git -t adversarial
docker run -it -p 8888:8888 --gpus '"device=0"' --mount type=bind,source=/mnt/savefiles/sda/josueortc/adversarial_notebook,destination=/notebooks/local_notebooks --mount type=bind,source=/home/josueortc/adversarial-docker/notebooks/,destination=/notebooks/premade_notebooks adversarial


```

Need to select:
- port <some number>:8888
Error: "docker: Error response from daemon: driver failed programming external connectivity on endpoint objective_brown (5da41f088043ba448ffbf29a07d0ae7fb491442b288afcfb0798fdc0d2dbbe6b): Bind for 0.0.0.0:8888 failed: port is already allocated.
ERRO[0001] error waiting for container: context canceled"

Means you need to use another port number
-Device: check which GPUs are available
- Mount 1st: Where are you going to save your results. Mine are in the example directory (You can look for my files there).
- Mount 2nd: To load my notebooks

This will create a jupyter lab
