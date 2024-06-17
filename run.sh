#!/bin/bash
nvcc -I /usr/local/cuda/include/ \
 -I /usr/local/cuda/targets/x86_64-linux/lib/ \
 strrev_gds.cu \
 -o strrev_gds.co \
 -L /usr/local/cuda/targets/x86_64-linux/lib/ \
 -lcufile \
 -L /usr/local/cuda/lib64/ \
 -lcuda \
 -L   -Bstatic \
 -L /usr/local/cuda/lib64/\
  -lcudart_static \
  -lrt \
  -lpthread \
  -ldl 

blkzone reset /dev/nvme2n1
./strrev_gds.co /mnt/opt_weights/opt-125m.pt /mnt/ssd0/_opt_125m.pt
#./strrev_gds.co /mnt/opt_weights/opt-125m.pt /dev/nvme2n1
