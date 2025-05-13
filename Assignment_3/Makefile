$(shell mkdir -p exe)

# Compiler
CC = nvcc
# Compiler flags
INCLUDES = -I/cvmfs/sling.si/modules/el7/software/OpenMPI/4.1.6-GCC-13.2.0/build/include 
LDFLAGS = -L/cvmfs/sling.si/modules/el7/software/OpenMPI/4.1.6-GCC-13.2.0/build/lib/
LDLIBS = -lmpi -lm
CFLAGS = -O3 -Wno-deprecated-gpu-targets -Xcompiler --openmp,-Wall, 

# Source directory
SRC=src

# Source files
SRCS=$(wildcard $(SRC)/*mpi.cu)
SRCS+=$(wildcard $(SRC)/*mpi.c)

# Header files
HDRS = $(wildcard $(SRC)/*.h)

# Object files
OBJS_1 = $(SRCS:$(SRC)/%.c=exe/%.o)
OBJS = $(OBJS_1:$(SRC)/%.cu=exe/%.o)

# Executable name
TARGET = exe/grayscott_par_mpi

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

# Compile source files to object files
exe/%.o: src/%.cu $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

exe/%.o: src/%.c $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Clean
clean:
	rm -f $(OBJS) $(TARGET)
