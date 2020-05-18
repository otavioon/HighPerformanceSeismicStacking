SHELL = /bin/bash

#
# Compiler flags
#

ENABLE_DEBUG := yes
ifeq ($(ENABLE_DEBUG), yes)
	CFLAGS += -g -O0
	NVCCFLAGS += -G -g -O0
else
	CFLAGS += -O3
	NVCCFLAGS += -O3
endif

LDFLAGS += -lboost_program_options -lboost_system -lboost_filesystem -lm -lcuda
CFLAGS += -Wall -fPIC -std=c++14 -Wno-sign-conversion
NVCCFLAGS += --use_fast_math -arch=sm_30 -Xcompiler -Wall -Xcompiler -fPIC -lcurand

#
# Folder paths
#

PROJECT_ROOT_PATH = ..
BINARY_PATH = bin/
SINGLE_HOST_PATH = $(BINARY_PATH)/single_host

CPP_SOURCE = $(shell find $(PROJECT_ROOT_PATH)/{common,cuda}/src -name '*.cpp')
CPP_OBJECT = $(CPP_SOURCE:.cpp=.o)

CUDA_SOURCE = $(shell find $(PROJECT_ROOT_PATH)/cuda/src -name '*.cu')
CUDA_OBJECT = $(CUDA_SOURCE:.cu=.o)

CUDA_LIBRARY_PATH = /usr/local/cuda/include/

SPITS_LIBRARY_PATH = $(PROJECT_ROOT_PATH)/spitz/spitz-include/ccpp/

%.o: %.cu
	nvcc -o $@ -c $< -dc -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I$(SPITS_LIBRARY_PATH) $(NVCCFLAGS)

%.o: %.cpp
	g++ -o $@ -c $< -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(CFLAGS)

dependencies: $(CPP_OBJECT) $(CUDA_OBJECT);

single_host_linear_search: $(SINGLE_HOST_PATH)/linear_search.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	nvcc -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

single_host_de: $(SINGLE_HOST_PATH)/differential_evolution.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	nvcc -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

single_host_stretch_free: $(SINGLE_HOST_PATH)/stretch_free.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	nvcc -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

all: single_host_linear_search single_host_de single_host_stretch_free

clean:
	rm -rf $(CPP_OBJECT)
	rm -rf $(CUDA_OBJECT)
