FLAGS=-O3 -g
HOST_FLAGS=${FLAGS} -ftemplate-depth=512
BINDIR = bin
OBJDIR = obj

PROGS =  ${BINDIR}/fc_mnist ${BINDIR}/conv_mnist ${BINDIR}/alexnet ${BINDIR}/resnet34b ${BINDIR}/resnet50

all: ${PROGS} ${OBJDIR}/neural_net_graph.ppo
	@echo Build complete

${BINDIR}/:
	-mkdir ${BINDIR}

${OBJDIR}/:
	-mkdir ${OBJDIR}

NEURAL_NET_GRAPH_HEADERS = neural_net_common.h PerformanceEstimation.hpp

${OBJDIR}/neural_net_graph.ppo: neural_net_graph.cpp ${NEURAL_NET_GRAPH_HEADERS} Makefile | ${OBJDIR}/
	popc -I. ${FLAGS} $< -o $@

${OBJDIR}/%.o: %.cpp Makefile ${wildcard *.hpp} ${wilcard *.h} | ${OBJDIR}/
	${CXX} ${HOST_FLAGS} -c -std=c++11 $< -o $@

COMMON_OBJ = \
	${OBJDIR}/ActivationMapping.o \
	${OBJDIR}/Convolution.o \
	${OBJDIR}/ConvPlan.o \
        ${OBJDIR}/ConvReuse.o \
	${OBJDIR}/ConvUtil.o \
	${OBJDIR}/FullyConnected.o \
	${OBJDIR}/FullyConnectedPlan.o \
	${OBJDIR}/MaxPool.o \
	${OBJDIR}/Net.o \
	${OBJDIR}/NonLinearity.o

COMMON_LIBS = \
	-lpoplar \
	-lboost_program_options

ALEXNET_OBJ=${OBJDIR}/alexnet.o ${COMMON_OBJ}

${BINDIR}/alexnet: ${ALEXNET_OBJ} | ${BINDIR}/
	${CXX} -std=c++11 -ffast-math ${ALEXNET_OBJ} ${COMMON_LIBS} -o $@

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
	${CXX} -std=c++11 -ffast-math ${FC_MNIST_OBJ} ${COMMON_LIBS} -lpoplar -o $@

CONV_MNIST_OBJ=${OBJDIR}/conv_mnist.o ${OBJDIR}/mnist.o ${COMMON_OBJ}

${BINDIR}/conv_mnist: ${CONV_MNIST_OBJ} | ${BINDIR}/
	${CXX} -std=c++11 -ffast-math ${CONV_MNIST_OBJ} ${COMMON_LIBS} -lpoplar -o $@

clean:
	-rm ${OBJDIR}/* ${BINDIR}/*
	-rmdir ${OBJDIR}
	-rmdir ${BINDIR}

