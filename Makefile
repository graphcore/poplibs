FLAGS=-O3 -g
BINDIR = bin
OBJDIR = obj

#PROGS = ${BINDIR}/fc ${BINDIR}/fc_mnist ${BINDIR}/alexnet
PROGS =  ${BINDIR}/alexnet ${BINDIR}/resnet34b ${BINDIR}/resnet50

all: ${PROGS} ${OBJDIR}/neural_net_graph32.ppo ${OBJDIR}/neural_net_graph16.ppo
	@echo Build complete

${BINDIR}/:
	-mkdir ${BINDIR}

${OBJDIR}/:
	-mkdir ${OBJDIR}

NEURAL_NET_GRAPH_HEADERS = neural_net_common.h PerformanceEstimation.hpp

${OBJDIR}/neural_net_graph16.ppo: neural_net_graph.cpp ${NEURAL_NET_GRAPH_HEADERS} Makefile | ${OBJDIR}/
	popc -DFPType=short ${FLAGS} $< -o $@

${OBJDIR}/neural_net_graph32.ppo: neural_net_graph.cpp ${NEURAL_NET_GRAPH_HEADERS} Makefile | ${OBJDIR}/
	popc -DFPType=float ${FLAGS} $< -o $@

${OBJDIR}/%.o: %.cpp Makefile ${wildcard *.hpp} ${wilcard *.h} | ${OBJDIR}/
	${CXX} -march=native -ffast-math ${FLAGS} -c -std=c++11 $< -o $@

COMMON_OBJ = \
	${OBJDIR}/ConvLayer.o \
	${OBJDIR}/FullyConnectedLayer.o \
	${OBJDIR}/Layer.o \
	${OBJDIR}/MaxPoolLayer.o \
	${OBJDIR}/Net.o

COMMON_LIBS = \
	-lpoplar \
	-lboost_program_options

ALEXNET_OBJ=${OBJDIR}/alexnet.o ${COMMON_OBJ}

${BINDIR}/alexnet: ${ALEXNET_OBJ} | ${BINDIR}/
	${CXX} -v -std=c++11 -ffast-math ${ALEXNET_OBJ} ${COMMON_LIBS} -o $@

RESNET34B_OBJ=${OBJDIR}/resnet34b.o ${COMMON_OBJ}

${BINDIR}/resnet34b: ${RESNET34B_OBJ} | ${BINDIR}/
	${CXX} -std=c++11 -ffast-math ${RESNET34B_OBJ} ${COMMON_LIBS} -o $@

RESNET50_OBJ=${OBJDIR}/resnet50.o ${COMMON_OBJ}

${BINDIR}/resnet50: ${RESNET50_OBJ} | ${BINDIR}/
	${CXX} -std=c++11 -ffast-math ${RESNET50_OBJ} ${COMMON_LIBS} -o $@

FC_OBJ=${OBJDIR}/fc.o ${COMMON_OBJ}

${BINDIR}/fc: ${FC_OBJ} | ${BINDIR}/
	${CXX} -std=c++11 -ffast-math ${FC_OBJ} -lpoplar -o $@

FC_MNIST_OBJ=${OBJDIR}/fc_mnist.o ${OBJDIR}/mnist.o ${COMMON_OBJ}

${BINDIR}/fc_mnist: ${FC_MNIST_OBJ} | ${BINDIR}/
	${CXX} -std=c++11 -ffast-math ${FC_MNIST_OBJ} -lpoplar -o $@

clean:
	-rm ${OBJDIR}/* ${BINDIR}/*
	-rmdir ${OBJDIR}
	-rmdir ${BINDIR}

