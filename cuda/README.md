# CUDA

CUDA specific source code to process seismic data.

## Building

A Makefile is provided in order to generate all available binaries. Before starting, assure yourself to set up your environment with the latest CUDA libraries and APIs.
After doing that, locate the CUDA's header location (usually in **/usr/local/cuda/include/**) and edit *CUDA_LIBRARY_PATH* and *NVCC* variables inside this Makefile.
Also check the compute capabilities of your graphic card and use **ARCH=sm_30** when calling **make**.

Six differents binaries might be compiled by the Makefile:

* **single_host_linear_search**: single host implementation for linear (greedy) search.
* **single_host_de**: single host implementation for differential evolution based search.
* **single_host_stretch_free**: to be used after the parameters have been computed. It stackes the data assuring the final result is stretch free.
* **spitz_linear_search***: spits implementation for linear (greedy) search.
* **spitz_de***: spits implementation for differential evolution based search.

Running **make all ARCH=<arch>** builds them all and **make clean** cleans up your workspace.
