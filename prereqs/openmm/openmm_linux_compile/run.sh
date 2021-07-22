#!/bin/bash
singularity run --fakeroot --bind .:/host centos-7-devtoolset-9-py39-cuda-openmm.sif
