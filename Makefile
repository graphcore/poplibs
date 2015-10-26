FLAGS=-O3 -g
BINDIR = bin
OBJDIR = obj

all: ${BINDIR}/nn_mnist ${BINDIR}/nn_imagenet ${OBJDIR}/neural_net_graph.ppo
	@echo Build complete

${BINDIR}/:
	-mkdir ${BINDIR}

${OBJDIR}/:
	-mkdir ${OBJDIR}

${OBJDIR}/neural_net_graph.ppo: neural_net_graph.cpp neural_net_common.h Makefile | ${OBJDIR}/
	popc ${FLAGS} $< -o $@

${OBJDIR}/%.o: %.cpp Makefile ${wildcard *.hpp} ${wilcard *.h} | ${OBJDIR}/
	g++ -march=native -ffast-math ${FLAGS} -c -std=c++11 $< -o $@

NN_MNIST_OBJ=${OBJDIR}/nn_mnist.o ${OBJDIR}/mnist.o

${BINDIR}/nn_mnist: ${NN_MNIST_OBJ} | ${BINDIR}/
	g++ -std=c++11 -ffast-math ${NN_MNIST_OBJ} -lpoplar -o $@

NN_IMAGENET_OBJ=${OBJDIR}/nn_imagenet.o

${BINDIR}/nn_imagenet: ${NN_IMAGENET_OBJ} | ${BINDIR}/
	g++ -std=c++11 -ffast-math ${NN_IMAGENET_OBJ} -lpoplar -o $@

clean:
	-rm ${OBJDIR}/* ${BINDIR}/*
	-rmdir ${OBJDIR}
	-rmdir ${BINDIR}

