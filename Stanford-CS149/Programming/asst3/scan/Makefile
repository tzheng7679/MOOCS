EXECUTABLE := cudaScan

CU_FILES   := scan.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

ARCH := $(shell uname -m)

# set default clean and dirs commands
MKDIRS = mkdir -p $(OBJDIR)/
CLEANS = rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

ifeq ($(ARCH), aarch64)
    CXX = g++
else ifeq ($(ARCH), )
    # building on windows
    LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lcudart -lstdc++ -g
    # edit dirs and clean commands
    MKDIRS=if not exist $(OBJDIR) mkdir $(OBJDIR)
    CLEANPATHS='$(OBJDIR)', '*.ppm', '*~', '$(EXECUTABLE)', '$(LOGS)'
    CLEANS = powershell.exe -Command "$$paths = $(CLEANPATHS); foreach($$path in $$paths) { if(Test-Path $$path) {rm -force -r $$path} }"
else
   CXX = g++ -m64
endif

OBJDIR=objs
CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/scan.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		$(MKDIRS)

clean:
		$(CLEANS)

check_scan: default
				./checker.py scan

check_find_repeats: default
				./checker.py find_repeats

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
