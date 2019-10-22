#ifndef poplibs_support_StructHelper_hpp
#define poplibs_support_StructHelper_hpp

#include <tuple>

namespace poplibs_support {

// generic helper struct that provides some reflection like methods for
// any object provided it is constructed with a list of pointers to
// data members for that object.
template <typename... Ps> struct StructHelper {
  std::tuple<Ps...> members;

  constexpr explicit StructHelper(Ps &&... ps)
      : members{std::forward<Ps>(ps)...} {}

  template <typename T> constexpr bool lt(const T &lhs, const T &rhs) const {
    return tie(lhs) < tie(rhs);
  }

  template <typename T> constexpr bool eq(const T &lhs, const T &rhs) const {
    return tie(lhs) == tie(rhs);
  }

private:
  template <typename T> constexpr auto tie(const T &t) const {
    return tie(t, std::make_index_sequence<sizeof...(Ps)>{});
  }

  template <typename T, std::size_t... Is>
  constexpr auto tie(const T &t, std::index_sequence<Is...>) const {
    const auto get = [&t](const auto &member) -> const auto & {
      return t.*member;
    };

    return std::tie(get(std::get<Is>(members))...);
  }
};

template <typename... Ps>
constexpr StructHelper<Ps...> makeStructHelper(Ps &&... ps) {
  return StructHelper<Ps...>{std::forward<Ps>(ps)...};
}

} // namespace poplibs_support

#endif // poplibs_support_StructHelper_hpp
