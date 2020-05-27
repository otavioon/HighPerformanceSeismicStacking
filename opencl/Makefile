ENABLE_DEBUG := no
ifeq ($(ENABLE_DEBUG), yes)
	CFLAGS += -g -O0
else
	CFLAGS += -O3
endif

CFLAGS += -Wconversion

LDFLAGS += -lm -lOpenCL -pthread

COMMON_FLDR ?= ../common
BIN_FLDR ?= bin

common_include = ${COMMON_FLDR}/include/
common_src = $(wildcard ${COMMON_FLDR}/src/*.cpp)
common_obj = $(common_src:.cpp=.o)

include = include/
src = $(wildcard src/*.cpp)
obj = $(src:.cpp=.o)

SPITZ_FLDR ?= ../spitz
spitz_include = $(SPITZ_FLDR)/spitz-include/ccpp/
CFLAGS += -fPIC

%.o: %.cpp
	g++ -o $@ -c $< -I${common_include} -I${include} -I${spitz_include} $(CFLAGS)

cmp-crs-crp: cmp-crs-crp.cpp $(obj) $(common_obj)
	mkdir -p ${BIN_FLDR}
	g++ -o  ${BIN_FLDR}/$@ -I${common_include} -I${include} $^ $(LDFLAGS) $(CFLAGS)

cmp-crs-crp-ga: cmp-crs-crp-ga.cpp $(obj) $(common_obj)
	mkdir -p ${BIN_FLDR}
	g++ -o ${BIN_FLDR}/$@ -I${common_include} -I${include} $^ $(LDFLAGS) $(CFLAGS)

stack: stack.cpp $(obj) $(common_obj)
	mkdir -p ${BIN_FLDR}
	g++ -o ${BIN_FLDR}/$@ -I${common_include} -I${include} $^ $(LDFLAGS) $(CFLAGS)

spitz: spitz.cpp $(obj) $(common_obj)
	mkdir -p ${BIN_FLDR}
	g++ -o ${BIN_FLDR}/$@ -I${common_include} -I${include} -I${spitz_include} $^ $(LDFLAGS) $(CFLAGS) -shared

spitz-ga: spitz-ga.cpp $(obj) $(common_obj)
	mkdir -p ${BIN_FLDR}
	g++ -o ${BIN_FLDR}/$@ -I${common_include} -I${include} -I${spitz_include} $^ $(LDFLAGS) $(CFLAGS) -shared

spitz-stack: spitz-stack.cpp $(obj) $(common_obj)
	mkdir -p ${BIN_FLDR}
	g++ -o ${BIN_FLDR}/$@ -I${common_include} -I${include} -I${spitz_include} $^ $(LDFLAGS) $(CFLAGS) -shared

all: cmp-crs-crp cmp-crs-crp-ga spitz spitz-ga stack spitz-stack

clean:
	rm -rf ${BIN_FLDR}/
	rm -rf $(obj) $(common_obj)
