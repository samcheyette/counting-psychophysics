# Define where Fleet lives (directory containing src)
FLEET_ROOT=../Fleet

include $(FLEET_ROOT)/Fleet.mk

CXX ?= g++

LOCAL_FLEET_FLAGS := $(FLEET_FLAGS)
ifeq ($(shell $(CXX) --version 2>/dev/null | head -n1 | grep -ci clang),1)
LOCAL_FLEET_FLAGS := $(filter-out -fvar-tracking-assignments --param max-inline-insns-recursive=100000,$(LOCAL_FLEET_FLAGS))
endif

all:
	$(CXX) -o main Main.cpp -O2 $(LOCAL_FLEET_FLAGS) $(FLEET_INCLUDE) $(FLEET_LIBS)

debug:
	$(CXX) -o main Main.cpp -g -O2 $(LOCAL_FLEET_FLAGS) $(FLEET_INCLUDE) $(FLEET_LIBS)

clang:
	clang++ -o main Main.cpp -O2 $(CLANG_FLAGS) $(FLEET_INCLUDE) $(FLEET_LIBS)
