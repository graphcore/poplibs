FLAGS=-O3 -g -DUSE_GATHER_VERTEX=1 -DMEM_OPTIMIZED_WEIGHT_SYNC=1 -DIPU_MODEL=0 -DLARGE_DNN_MODEL=0 -DDO_COMPUTATION=1 -DTRAIN_SINGLE_BATCH_ONLY=0

all: nn neural_net_graph.ppo
	@echo Build complete

neural_net_graph.ppo: neural_net_graph.cpp
	popc ${FLAGS} $< -o $@

%.o: %.cpp
	g++ -march=native -ffast-math ${FLAGS} -c -std=c++11 $< -o $@

OBJ=neural_net_host.o mnist.o

nn: ${OBJ}
	g++ -std=c++11 -ffast-math ${OBJ} -lpoplar -o nn

clean:
	-rm *.o *.ppo nn

