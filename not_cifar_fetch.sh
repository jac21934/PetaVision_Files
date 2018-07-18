#!/bin/bash

# Download 480x320 CIFAR-like image set

fileid="1IVGOo0GDBl3I_R1Krlllc2AMAb-vAKyq"
filename="not_cifar.tar.gz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
