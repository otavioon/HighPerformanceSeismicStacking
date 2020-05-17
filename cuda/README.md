# CUDA

NVIDIA specific source code to process seismic data.

## Building

A Makefile is provided in order to generate all available binaries. Before starting, assure yourself to set up your environment with the latest CUDA libraries and APIs.
After doing that, locate the CUDA's header location (usually in **/usr/local/cuda/include/**) and edit *cuda_include* variable inside Makefile.
Also check the compute capabilities of your graphic card and replace **-arch=sm_30** in this same file accordingly.

Six differents binaries might be compiled by the Makefile:

* **cmp-crs-crp**: best parameters for each (m0, t0)-tuple are search linearly over a finite range.
* **cmp-crs-crp-ga**: the search for the parameters is realized by a differential evolution strategy.
* **stack**: to be used after the parameters have been computed. It stackes the data assuring the final result is stretch free.
* **spitz-***: methods described above, but compatible with Spits.

Running **make all** builds them all and **make clean** cleans up your workspace. 

## Running

A helper script is provided in the **tests** folder to help you running the binaries for a sample data. Please refer to the instructions in that folder.

You can also execute `$ ./bin/cmp-crs-crp -h` or `$ ./bin/cmp-crs-crp-ga -h` to list all the available options.

For Spitz-related questions, please refer to its own documentation for further information.

