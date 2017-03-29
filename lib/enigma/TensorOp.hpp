#ifndef __TensorOp_hpp__
#define __TensorOp_hpp__
#include <boost/fusion/container.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/functional.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/reverse.hpp>
#include <boost/optional.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <map>

/** This header provides an interface for resusable 'tensor operations' i.e.
 *  functions that provide a control program to manipulate some tensors.
 *
 *  Reuse is performed by creating the program to manipulate extra
 *  tensors that act as parameters. When the operation is used on argument
 *  tensors input tensors are copied in to the parameters, the program is run
 *  and then the any outputs tensors are copied out.
 *
 *  An operation is created for a function that returns a program type.
 *  The operation will create a map to resuable programs based on the arguments
 *  supplied to this function. Optionally, some arguments can be marked
 *  as non-indexable. The tensor arguments will match a prior call if the
 *  dimensions and element type are the same.
 *
 *  Each tensor parameter to the underlying function needs to be marked as
 *  an input, output or inout parameter to direct the copying. This is
 *  doing when calling the 'createTensorOp' function. Additional an optional
 *  tile mapping function can be provided to map the parameter tensor if the
 *  underlying function does not do this.
 */

namespace popnn {

// Specification of tensor parameter behaviour
enum class TensorOpParamType {
  InputTensor,
  OutputTensor,
  InOutTensor,
  NotParamTensor
};

template <unsigned Index>
class TensorOpArg {};

static const std::map<TensorOpParamType, std::string>
paramTypeName{{TensorOpParamType::InputTensor, "input"},
              {TensorOpParamType::OutputTensor, "output"},
              {TensorOpParamType::InOutTensor, "inout"}};

using MappingFunction =
    std::function<void(poplar::Graph &, const poplar::Tensor&)>;

struct OpParamDesc {
  TensorOpParamType type;
  boost::optional<MappingFunction> mappingFn;
  OpParamDesc(TensorOpParamType type,
              MappingFunction mappingFn) : type(type), mappingFn(mappingFn) {}
  OpParamDesc(TensorOpParamType type) : type(type) {}
};

using OpSignature = std::vector<OpParamDesc>;

template <typename NonIndexArgs, typename ...Args>
struct TensorOp {

  using TensorShape = std::pair<std::vector<std::size_t>, std::string>;

  // This operation converted function arguments into keys for a
  // map between arguments and control programs. Tensors are
  // converted to their dimensions and element types. The
  // poplar::Graph type and any argument index encoded in the NonIndexArgs list
  // are ignored by transforming them into integer constant zero.
  struct MakeKey {
    poplar::Graph &graph;

    template <unsigned I>
    using IsNotKey = boost::mpl::contains<NonIndexArgs, TensorOpArg<I>>;
    template<unsigned I>
    using EnableIfNotKey = typename boost::enable_if<IsNotKey<I>>::type;
    template<unsigned I>
    using EnableIfKey = typename
        boost::enable_if<boost::mpl::not_<IsNotKey<I>>>::type;
    template <typename V, typename T>
    using pb = typename boost::fusion::result_of::push_back<const V, T>::type;

    template <typename V, typename T, unsigned I>
    std::pair<TensorOpArg<I + 1>, pb<V, T>>
    operator()(const std::pair<TensorOpArg<I>, V> &p,
               const T &x, EnableIfKey<I>* = 0) const {
      return {TensorOpArg<I + 1>(), boost::fusion::push_back(p.second, x)};
    }

    template <typename V, typename T, unsigned I>
    std::pair<TensorOpArg<I + 1>, pb<V, int>>
    operator()(const std::pair<TensorOpArg<I>, V> &p,
               const T &, EnableIfNotKey<I>* = 0) const {
      return {TensorOpArg<I + 1>(), boost::fusion::push_back(p.second, 0)};
    }

    template <typename V, unsigned I>
    std::pair<TensorOpArg<I + 1>, pb<V, int>>
    operator()(const std::pair<TensorOpArg<I>, V> &p,
               const poplar::Graph &) const {
      return {TensorOpArg<I + 1>(), boost::fusion::push_back(p.second, 0)};
    }

    template <typename V, unsigned I>
    std::pair<TensorOpArg<I + 1>, pb<V, TensorShape>>
    operator()(const std::pair<TensorOpArg<I>, V> &p,
               const poplar::Tensor &t) const {
      return {TensorOpArg<I + 1>(),
              boost::fusion::push_back(
               p.second,
               TensorShape(t.shape(), graph.getTensorElementType(t))
             )};
    }

    MakeKey(poplar::Graph &graph) : graph(graph) {}
  };

  // The key type for the program map is the result of applying
  // transforming and applying MakeKey to each element. The end type
  // is converted to a list to force evaluation of the type.
  using KeyType =
    typename boost::fusion::result_of::as_list<
      typename boost::fusion::result_of::fold<
         boost::fusion::vector<Args...>,
         std::pair<TensorOpArg<0>, boost::fusion::vector<>>,
         MakeKey
      >::type::second_type
    >::type;

  using TensorParams = std::vector<poplar::Tensor>;
  using ProgAndParams = std::pair<poplar::program::Program, TensorParams>;

  std::map<KeyType, ProgAndParams> cache;

  poplar::Graph &graph;
  poplar::program::Program (*fn)(Args...);
  std::string name;
  OpSignature sig;

  struct ArgCopier {
    poplar::Graph &graph;
    const TensorParams &params;
    const OpSignature &sig;
    poplar::program::Sequence &inputCopies;
    poplar::program::Sequence &outputCopies;
    unsigned &tIndex;
    void operator()(const poplar::Tensor &t) const {
      const auto &param = params[tIndex];
      const auto opType = sig[tIndex].type;
      if (opType == TensorOpParamType::InputTensor ||
          opType == TensorOpParamType::InOutTensor) {
        inputCopies.add(poplar::program::Copy(t, param));
      }
      if (opType == TensorOpParamType::OutputTensor ||
          opType == TensorOpParamType::InOutTensor) {
        outputCopies.add(poplar::program::Copy(param, t));
      }
      ++tIndex;
    }
    template <typename T>
    void operator()(const T &x) const { }
    ArgCopier(poplar::Graph &graph, const TensorParams &params,
              const OpSignature &sig, poplar::program::Sequence &inputCopies,
              poplar::program::Sequence &outputCopies,
              unsigned &tIndex)
      : graph(graph), params(params),
        sig(sig), inputCopies(inputCopies), outputCopies(outputCopies),
        tIndex(tIndex) {
      tIndex = 0;
    }
  };

  // This type is used as a dummy argument in the run method to
  // provide a base for the recursion through function arguments via
  // type specialization.
  class Sentinel{};

  template <typename F, typename ...FArgs>
  poplar::program::Program run(unsigned, TensorParams &, F f,
                               Sentinel, FArgs&&... args) {
    return f(std::forward<FArgs>(args)...);
  }

  template <typename F, typename FArg, typename ...FArgs>
  poplar::program::Program run(unsigned tIndex, TensorParams &params,
                               F f, FArg &&arg, FArgs&&... args) {
    return run(tIndex, params, f,
               std::forward<FArgs>(args)...,
               std::forward<FArg>(arg));
  }

  template <typename F, typename ...FArgs>
  poplar::program::Program run(unsigned tIndex, TensorParams &params, F f,
                               poplar::Tensor &t, FArgs&&... args) {
    auto paramType = sig[tIndex].type;
    if (paramType == TensorOpParamType::NotParamTensor) {
      params.push_back(t);
      return run(tIndex + 1, params, f, std::forward<FArgs>(args)..., t);
    }
    auto paramTypeStr = paramTypeName.find(paramType)->second;
    auto paramName = name + "." + paramTypeStr + "." + std::to_string(tIndex);
    auto param = graph.addTensor(graph.getTensorElementType(t), t.shape(),
                                 paramName);
    const auto &mappingFn = sig[tIndex].mappingFn;
    if (mappingFn)
      (*mappingFn)(graph, param);
    params.push_back(param);
    return run(tIndex + 1, params, f, std::forward<FArgs>(args)..., param);
  }

  poplar::program::Program
  operator()(Args... args) {
    auto vArgs = boost::fusion::vector_tie(args...);
    auto key = boost::fusion::fold(vArgs,
                                   std::make_pair(TensorOpArg<0>(),
                                                  boost::fusion::make_vector()),
                                   MakeKey(graph)).second;
    const auto match = cache.find(key);
    const ProgAndParams *p;
    if(match == cache.end()) {
      TensorParams tensorParams;
      auto prog = run(0, tensorParams, fn, args..., Sentinel());
      ProgAndParams pp = std::make_pair(std::move(prog),
                                        std::move(tensorParams));
      auto it = cache.emplace(std::move(key), std::move(pp));
      p = &it.first->second;
    } else {
      p = &match->second;
    }
    auto inputCopies = poplar::program::Sequence();
    auto outputCopies = poplar::program::Sequence();
    unsigned tIndex;
    boost::fusion::for_each(vArgs,
                            ArgCopier(graph, p->second, sig,
                                      inputCopies, outputCopies,
                                      tIndex));
    return poplar::program::Sequence(inputCopies, p->first, outputCopies);
  }
  TensorOp(poplar::Graph &graph,
           poplar::program::Program (*fn)(Args...),
           std::string name,
           OpSignature sig) :
    graph(graph), fn(fn), name(std::move(name)), sig(std::move(sig)) {}
};


template <unsigned... NonIndexArgs, typename... Args>
static
TensorOp<boost::fusion::vector<TensorOpArg<NonIndexArgs>...>, Args...>
createTensorOp(poplar::Graph &graph,
               poplar::program::Program (*fn)(Args...),
               std::string name,
               OpSignature sig) {
  return
    TensorOp<boost::fusion::vector<TensorOpArg<NonIndexArgs>...>,
             Args...>(
        graph, fn, std::move(name),  std::move(sig)
    );
}

} // end namespace popnn

#define POPNN_TENSOR_OP_TYPE(x, NonIndexArgs...) \
  decltype( \
    popnn::createTensorOp<NonIndexArgs>( \
      std::declval<poplar::Graph&>(), \
      x, \
      std::declval<std::string>(), \
      std::declval<popnn::OpSignature>() \
    ) \
  )

#endif // __TensorOp_hpp__
