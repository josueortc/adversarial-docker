FROM bethgelab/deeplearning:cuda9.0-cudnn7

RUN pip3 install -U setuptools
RUN pip3 install --upgrade pip==19.0.1
RUN pip3 install tqdm foolbox eagerpy==0.15.0
RUN pip3 install foolbox-native

#RUN echo "c.NotebookApp.token = u''" >> /usr/.jupyter/jupyter_notebook_config.py
#COPY notebook.json /usr/.Jupiter/nbconfig
RUN pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numba jax jaxlib
RUN pip3 install ruamel.yaml
RUN pip3 install imageio
RUN pip3 install imageio-ffmpeg

RUN pip3 install numpy==1.17.0
#RUN pip3 install scipy==1.1.0
RUN pip3 install opencv-python
# Export port for Jupyter Notebook
EXPOSE 8888

# Add Jupyter Notebook config
ADD ./jupyter_notebook_config.py /root/.jupyter/

WORKDIR /notebooks

# By default start running jupyter notebook
ENTRYPOINT ["jupyter", "lab", "--allow-root"]
