# All-in-one Jupyter Docker image for GPU Deep Learning using PyCUDA, ArrayFire, CUDA etc. 

NOTE: Building this image may take several hours since CMAKE is being built from source. 
https://github.com/QuantScientist/deep-ml-meetups


# Image contents
On top of all the fancy deep learning libraries, this docker image contains:

* ArrayFire
* PyCUDA
* Python 
* LLVM
* LLDB
* Snappy
* Numba

 
# Build the image

docker build -t quantscientist/gpu -f Dockerfile.cpu .

# Run the image
docker run -it -p 5555:5555 -p 7842:7842 -p 8787:8787 -p 8786:8786 -p 8788:8788 -v /myhome/data-science/:/root/sharedfolder  --env="DISPLAY"  --env="QT_X11_NO_MITSHM=1"  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  quantscientist/gpu bash


# Run Jupyter
chmod +x run_jupyter.sh
./run_jupyter.sh
![Jup](start.png)


OR

docker build -t quantscientist/gpu -f Dockerfile.gpu .


