MXNet CPU installation in R - Manjaro Linux/Archlinux
================

Document references: - [MXNet Getting
started](https://mxnet.incubator.apache.org/get_started/install.html)

This document provides instructions on how to install MXNet for R on CPU
mode for Manjaro or Archlinux based systems. It can be extrapolated to
other linux distros, however the commands will vary.

TODO:

  - Create guide for Ubuntu.
  - Create guide for Windows.

<!-- end list -->

    yaourt -S openblas-lapack
    sudo pacman -S opencv

Be patient with the following steps, they take a
    while:

    git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet --branch 0.11.0
    cd mxnet
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas

    make rpkg
    R CMD INSTALL mxnet_current_r.tar.gz

Test installation:

``` r
library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.cpu())
b <- a * 2 + 1
b
```

    ##      [,1] [,2] [,3]
    ## [1,]    3    3    3
    ## [2,]    3    3    3
