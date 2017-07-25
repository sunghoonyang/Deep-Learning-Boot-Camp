#!/usr/bin/env bash
nvidia-docker run -it -p 5555:5555 -p 7842:7842 -p 8787:8787 -p 8786:8786 -p 8788:8788 -v ~/db/Dropbox/dev2/:/root/sharedfolder  quantscientist/pycuda bash
