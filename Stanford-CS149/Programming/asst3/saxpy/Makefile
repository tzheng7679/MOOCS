EXECUTABLE := cudaSaxpy

CU_FILES   := saxpy.cu

CU_DEPS    :=

CC_FILES   := main.cpp

###########################################################
ARCH := $(shell uname -m)

ifeq ($(ARCH), aarch64)
    CXX = g++
else 
    CXX = g++ -m64
endif

# set default clean and dirs commands
MKDIRS = mkdir -p $(OBJDIR)/
CLEANS = rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

OBJDIR=objs
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart -g
else ifeq ($(ARCH), )
	# building on windows
	LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lcudart -lstdc++ -g
	# edit dirs and clean commands
	MKDIRS = if not exist $(OBJDIR) mkdir $(OBJDIR)
	CLEANPATHS = '$(OBJDIR)', '*.ppm', '*~', '$(EXECUTABLE)'
	CLEANS = powershell.exe -Command "$$paths = $(CLEANPATHS); foreach($$path in $$paths) { if(Test-Path $$path) {rm -force -r $$path} }"
else
# Building on Linux
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart -g
endif

NVCC=nvcc
NVCCFLAGS=-O3 -m64


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/saxpy.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		$(MKDIRS)

clean:
		$(CLEANS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@