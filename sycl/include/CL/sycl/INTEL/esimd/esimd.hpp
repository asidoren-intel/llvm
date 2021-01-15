//==------------ - esimd.hpp - DPC++ Explicit SIMD API   -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD vector APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/INTEL/esimd/detail/esimd_intrin.hpp>
#include <CL/sycl/INTEL/esimd/detail/esimd_types.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace INTEL {
namespace gpu {

/// The simd vector class.
///
/// This is a wrapper class for llvm vector values. Additionally this class
/// supports region operations that map to Intel GPU regions. The type of
/// a region select or format operation is of simd_view type, which models
/// read-update-write semantics.
///
/// \ingroup sycl_esimd
template <typename Ty, int N> class simd {
public:
  /// The underlying builtin data type.
  using vector_type = vector_type_t<Ty, N>;

  /// The element type of this simd object.
  using element_type = Ty;

  /// The number of elements in this simd object.
  static constexpr int length = N;

  // TODO @rolandschulz
  // Provide examples why constexpr is needed here.
  //
  /// @{
  /// Constructors.
  constexpr simd() = default;
  constexpr simd(const simd &other) { set(other.data()); }
  constexpr simd(simd &&other) { set(other.data()); }
  constexpr simd(const vector_type &Val) { set(Val); }

  // TODO @rolandschulz
  // {quote}
  // Providing both an overload of initializer-list and the same type itself
  // causes really weird behavior. E.g.
  //   simd s1(1,2); //calls next constructor
  //   simd s2{1,2}; //calls this constructor
  // This might not be confusing for all users but to everyone using
  // uniform-initialization syntax. Therefore if you want to use this
  // constructor the other one should have a special type (see
  // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es64-use-the-tenotation-for-construction)
  // to avoid this issue. Also this seems like one of those areas where this
  // simd-type needless differs from std::simd. Why should these constructors be
  // different? Why reinvent the wheel and have all the work of fixing these
  // problems if we could just use the existing solution. Especially if that is
  // anyhow the long-term goal. Adding extra stuff like the select is totally
  // fine. But differ on things which have no apparent advantage and aren't as
  // thought through seems to have only downsides.
  // {/quote}

  constexpr simd(std::initializer_list<Ty> Ilist) noexcept {
    int i = 0;
    for (auto It = Ilist.begin(); It != Ilist.end() && i < N; ++It) {
      M_data[i++] = *It;
    }
  }

  /// Initialize a simd with an initial value and step.
  constexpr simd(Ty Val, Ty Step = Ty()) noexcept {
    if (Step == Ty())
      M_data = Val;
    else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        M_data[i] = Val;
        Val += Step;
      }
    }
  }
  /// @}

  operator const vector_type &() const & { return M_data; }
  operator vector_type &() & { return M_data; }

  vector_type data() const {
#ifndef __SYCL_DEVICE_ONLY__
    return M_data;
#else
    return __esimd_vload<Ty, N>(&M_data);
#endif
  }

  /// Whole region read.
  simd read() const { return data(); }

  /// Whole region write.
  simd &write(const simd &Val) {
    set(Val.data());
    return *this;
  }

  /// Whole region update with predicates.
  void merge(const simd &Val, const mask_type_t<N> &Mask) {
    set(__esimd_wrregion<element_type, N, N, 0 /*VS*/, N, 1, N>(
        data(), Val.data(), 0, Mask));
  }
  void merge(const simd &Val1, simd Val2, const mask_type_t<N> &Mask) {
    Val2.merge(Val1, Mask);
    set(Val2.data());
  }

  /// {@
  /// Assignment operators.
  constexpr simd &operator=(const simd &) & = default;
  constexpr simd &operator=(simd &&) & = default;
  /// @}

  /// View this simd object in a different element type.
  template <typename EltTy> auto format() & {
    using TopRegionTy = compute_format_type_t<simd, EltTy>;
    using RetTy = simd_view<simd, TopRegionTy>;
    TopRegionTy R(0);
    return RetTy{*this, R};
  }

  // TODO @Ruyk, @iburyl - should renamed to bit_cast similar to std::bit_cast.
  //
  /// View as a 2-dimensional simd_view.
  template <typename EltTy, int Height, int Width> auto format() & {
    using TopRegionTy = compute_format_type_2d_t<simd, EltTy, Height, Width>;
    using RetTy = simd_view<simd, TopRegionTy>;
    TopRegionTy R(0, 0);
    return RetTy{*this, R};
  }

  /// 1D region select, apply a region on top of this LValue object.
  ///
  /// \tparam Size is the number of elements to be selected.
  /// \tparam Stride is the element distance between two consecutive elements.
  /// \param Offset is the starting element offset.
  /// \return the representing region object.
  template <int Size, int Stride>
  simd_view<simd, region1d_t<Ty, Size, Stride>> select(uint16_t Offset = 0) & {
    region1d_t<Ty, Size, Stride> Reg(Offset);
    return {*this, Reg};
  }

  /// 1D region select, apply a region on top of this RValue object.
  ///
  /// \tparam Size is the number of elements to be selected.
  /// \tparam Stride is the element distance between two consecutive elements.
  /// \param Offset is the starting element offset.
  /// \return the value this region object refers to.
  template <int Size, int Stride>
  simd<Ty, Size> select(uint16_t Offset = 0) && {
    simd<Ty, N> &&Val = *this;
    return __esimd_rdregion<Ty, N, Size, /*VS*/ 0, Size, Stride>(Val.data(),
                                                                 Offset);
  }

  // TODO
  // @rolandschulz
  // {quote}
  // - There is no point in having this non-const overload.
  // - Actually why does this overload not return simd_view.
  //   This would allow you to use the subscript operator to write to an
  //   element.
  // {/quote}
  /// Read a single element, by value only.
  Ty operator[](int i) const { return data()[i]; }

  // TODO
  // @rolandschulz
  // {quote}
  // - Why would the return type ever be different for a binary operator?
  // {/quote}
  //   * if not different, then auto should not be used
#define DEF_BINOP(BINOP, OPASSIGN)                                             \
  auto operator BINOP(const simd &RHS) const {                                 \
    using ComputeTy = compute_type_t<simd>;                                    \
    auto V0 = convert<typename ComputeTy::vector_type>(data());                \
    auto V1 = convert<typename ComputeTy::vector_type>(RHS.data());            \
    auto V2 = V0 BINOP V1;                                                     \
    return ComputeTy(V2);                                                      \
  }                                                                            \
  simd &operator OPASSIGN(const simd &RHS) {                                   \
    using ComputeTy = compute_type_t<simd>;                                    \
    auto V0 = convert<typename ComputeTy::vector_type>(data());                \
    auto V1 = convert<typename ComputeTy::vector_type>(RHS.data());            \
    auto V2 = V0 BINOP V1;                                                     \
    write(convert<vector_type>(V2));                                           \
    return *this;                                                              \
  }                                                                            \
  simd &operator OPASSIGN(const Ty &RHS) { return *this OPASSIGN simd(RHS); }

  // TODO @keryell
  // {quote}
  // Nowadays hidden friends seem to be more fashionable for these kind of
  // operations. A nice side effect is that you have easily some scalar
  // broadcast either on LHS & RHS.
  // {/quote}
  // TODO @mattkretz +1, ditto for compares
  DEF_BINOP(+, +=)
  DEF_BINOP(-, -=)
  DEF_BINOP(*, *=)
  DEF_BINOP(/, /=)

#undef DEF_BINOP

  // TODO @rolandschulz, @mattkretz
  // Introduce simd_mask type and let user use this type instead of specific
  // type representation (simd<uint16_t, N>) to make it more portable
  // TODO @iburyl should be mask_type_t, which might become more abstracted in
  // the future revisions.
  //
#define DEF_RELOP(RELOP)                                                       \
  simd<uint16_t, N> operator RELOP(const simd &RHS) const {                    \
    auto R = data() RELOP RHS.data();                                          \
    mask_type_t<N> M(1);                                                       \
    return M & convert<mask_type_t<N>>(R);                                     \
  }

  DEF_RELOP(>)
  DEF_RELOP(>=)
  DEF_RELOP(<)
  DEF_RELOP(<=)
  DEF_RELOP(==)
  DEF_RELOP(!=)

#undef DEF_RELOP

#define DEF_BITWISE_OP(BITWISE_OP, OPASSIGN)                                   \
  simd operator BITWISE_OP(const simd &RHS) const {                            \
    static_assert(std::is_integral<Ty>(), "not integeral type");               \
    auto V2 = data() BITWISE_OP RHS.data();                                    \
    return simd(V2);                                                           \
  }                                                                            \
  simd &operator OPASSIGN(const simd &RHS) {                                   \
    static_assert(std::is_integral<Ty>(), "not integeral type");               \
    auto V2 = data() BITWISE_OP RHS.data();                                    \
    write(convert<vector_type>(V2));                                           \
    return *this;                                                              \
  }

  DEF_BITWISE_OP(&, &=)
  DEF_BITWISE_OP(|, |=)
  DEF_BITWISE_OP(^, ^=)

#undef DEF_BITWISE_OP

#define DEF_LOGIC_OP(LOGIC_OP)                                                 \
  simd operator LOGIC_OP(const simd &RHS) const {                              \
    static_assert(std::is_integral<Ty>(), "not integeral type");               \
    auto V2 = data() LOGIC_OP RHS.data();                                      \
    return simd(convert<vector_type>(V2));                                     \
  }

  DEF_LOGIC_OP(&&)
  DEF_LOGIC_OP(||)
  //  DEF_LOGIC_OP(!, ^=)

#undef DEF_LOGIC_OP

  // Operator ++, --
  simd &operator++() {
    *this += 1;
    return *this;
  }
  simd operator++(int) {
    simd Ret(*this);
    operator++();
    return Ret;
  }
  simd &operator--() {
    *this -= 1;
    return *this;
  }
  simd operator--(int) {
    simd Ret(*this);
    operator--();
    return Ret;
  }

  /// \name Replicate
  /// Replicate simd instance given a region.
  /// @{
  ///

  /// \tparam Rep is number of times region has to be replicated.
  /// \return replicated simd instance.
  template <int Rep> simd<Ty, Rep * N> replicate() {
    return replicate<Rep, N>(0);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int W> simd<Ty, Rep * W> replicate(uint16_t Offset) {
    return replicate<Rep, W, W, 1>(Offset);
  }

  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W>
  simd<Ty, Rep * W> replicate(uint16_t Offset) {
    return replicate<Rep, VS, W, 1>(Offset);
  }

  // TODO
  // @rolandschulz
  // {quote}
  // - Template function with that many arguments are really ugly.
  //   Are you sure there isn't a better interface? And that users won't
  //   constantly forget what the correct order of the argument is?
  //   Some kind of templated builder pattern would be a bit more verbose but
  //   much more readable.
  //   ...
  //   The user would use (any of the extra method calls are optional)
  //   s.replicate<R>(i).width<W>().vstride<VS>().hstride<HS>()
  // {/quote}
  // @jasonsewall-intel +1 for this
  /// \tparam Rep is number of times region has to be replicated.
  /// \tparam VS vertical stride of src region to replicate.
  /// \tparam W is width of src region to replicate.
  /// \tparam HS horizontal stride of src region to replicate.
  /// \param Offset is offset in number of elements in src region.
  /// \return replicated simd instance.
  template <int Rep, int VS, int W, int HS>
  simd<Ty, Rep * W> replicate(uint16_t Offset) {
    return __esimd_rdregion<element_type, N, Rep * W, VS, W, HS, N>(
        data(), Offset * sizeof(Ty));
  }
  ///@}

  /// Any operation.
  ///
  /// \return 1 if any element is set, 0 otherwise.
  template <
      typename T1 = element_type, typename T2 = Ty,
      typename = sycl::detail::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t any() {
    return __esimd_any<Ty, N>(data());
  }

  /// All operation.
  ///
  /// \return 1 if all elements are set, 0 otherwise.
  template <
      typename T1 = element_type, typename T2 = Ty,
      typename = sycl::detail::enable_if_t<std::is_integral<T1>::value, T2>>
  uint16_t all() {
    return __esimd_all<Ty, N>(data());
  }

  /// Write a simd-vector into a basic region of a simd object.
  template <typename RTy>
  ESIMD_INLINE void writeRegion(
      RTy Region,
      const vector_type_t<typename RTy::element_type, RTy::length> &Val) {
    using ElemTy = typename RTy::element_type;
    if constexpr (N * sizeof(Ty) == RTy::length * sizeof(ElemTy))
      // update the entire vector
      set(bitcast<Ty, ElemTy, RTy::length>(Val));
    else {
      static_assert(!RTy::Is_2D);
      // If element type differs, do bitcast conversion first.
      auto Base = bitcast<ElemTy, Ty, N>(data());
      constexpr int BN = (N * sizeof(Ty)) / sizeof(ElemTy);
      // Access the region information.
      constexpr int M = RTy::Size_x;
      constexpr int Stride = RTy::Stride_x;
      uint16_t Offset = Region.M_offset_x * sizeof(ElemTy);

      // Merge and update.
      auto Merged = __esimd_wrregion<ElemTy, BN, M,
                                     /*VS*/ 0, M, Stride>(Base, Val, Offset);
      // Convert back to the original element type, if needed.
      set(bitcast<Ty, ElemTy, BN>(Merged));
    }
  }

  /// Write a simd-vector into a nested region of a simd object.
  template <typename TR, typename UR>
  ESIMD_INLINE void
  writeRegion(std::pair<TR, UR> Region,
              const vector_type_t<typename TR::element_type, TR::length> &Val) {
    // parent-region type
    using PaTy = typename shape_type<UR>::type;
    using ElemTy = typename TR::element_type;
    using BT = typename PaTy::element_type;
    constexpr int BN = PaTy::length;

    if constexpr (PaTy::Size_in_bytes == TR::Size_in_bytes) {
      writeRegion(Region.second, bitcast<BT, ElemTy, TR::length>(Val));
    } else {
      // Recursively read the base
      auto Base = readRegion<Ty, N>(data(), Region.second);
      // If element type differs, do bitcast conversion first.
      auto Base1 = bitcast<ElemTy, BT, BN>(Base);
      constexpr int BN1 = PaTy::Size_in_bytes / sizeof(ElemTy);

      if constexpr (!TR::Is_2D) {
        // Access the region information.
        constexpr int M = TR::Size_x;
        constexpr int Stride = TR::Stride_x;
        uint16_t Offset = Region.first.M_offset_x * sizeof(ElemTy);

        // Merge and update.
        Base1 = __esimd_wrregion<ElemTy, BN1, M,
                                 /*VS*/ 0, M, Stride>(Base1, Val, Offset);
      } else {
        static_assert(std::is_same<ElemTy, BT>::value);
        // Read columns with non-trivial horizontal stride.
        constexpr int M = TR::length;
        constexpr int VS = PaTy::Size_x * TR::Stride_y;
        constexpr int W = TR::Size_x;
        constexpr int HS = TR::Stride_x;
        constexpr int ParentWidth = PaTy::Size_x;

        // Compute the byte offset for the starting element.
        uint16_t Offset = static_cast<uint16_t>(
            (Region.first.M_offset_y * PaTy::Size_x + Region.first.M_offset_x) *
            sizeof(ElemTy));

        // Merge and update.
        Base1 = __esimd_wrregion<ElemTy, BN1, M, VS, W, HS, ParentWidth>(
            Base1, Val, Offset);
      }
      // Convert back to the original element type, if needed.
      auto Merged1 = bitcast<BT, ElemTy, BN1>(Base1);
      // recursively write it back to the base
      writeRegion(Region.second, Merged1);
    }
  }

private:
  // The underlying data for this vector.
  vector_type M_data;

  void set(const vector_type &Val) {
#ifndef __SYCL_DEVICE_ONLY__
    M_data = Val;
#else
    __esimd_vstore<Ty, N>(&M_data, Val);
#endif
  }
};

template <typename U, typename T, int n>
ESIMD_INLINE simd<U, n> convert(simd<T, n> val) {
  return __builtin_convertvector(val.data(), vector_type_t<U, n>);
}

//------------
// simd cf
//------------

// perfect forwarding

// return issue

// Chain issue. must be fixed. Otherwise it's not a if-else chain.
// if (a > 1) else if (a > 10) != if (a > 1) if (a > 10)

// constexpr

// Done
// in simd_if if conds.size == bodies.size -> it's the if-elif-elif. Otherwise,
// it's if-elif-else

namespace func {
template <typename Head, typename... Tail>
ESIMD_INLINE std::tuple<Tail...>
tuple_tail(const std::tuple<Head, Tail...> &t) {
  return std::apply(
      [](auto head, auto... tail) { return std::make_tuple(tail...); }, t);
}

template <typename... Conditions, typename... Bodies>
ESIMD_INLINE void simd_if_elif_else_impl(std::tuple<Conditions...> conds,
                                         std::tuple<Bodies...> bodies) {

  simd cond = std::get<0>(conds);
  if (__esimd_simdcf_any<typename decltype(cond)::element_type, cond.length>(
          cond)) {
    std::get<0>(bodies)();
    return;
  }
  simd_if_elif_else_impl(tuple_tail(conds), tuple_tail(bodies));
}

template <typename Body>
ESIMD_INLINE void simd_if_elif_else_impl(std::tuple<> empty_conds,
                                         std::tuple<Body> else_body) {
  std::get<0>(else_body)();
}

template <>
ESIMD_INLINE void simd_if_elif_else_impl(std::tuple<> empty_conds,
                                         std::tuple<> empty_body) {}

template <typename... Conditions, typename... Bodies>
ESIMD_INLINE void simd_if(std::tuple<Conditions...> conds,
                          std::tuple<Bodies...> bodies) {
  simd_if_elif_else_impl(conds, bodies);
}

} // namespace func

// std::optinal and void.

namespace cls {

template <typename T> static constexpr auto make_array(T &&t) {
  constexpr auto array_from_tuple = [](auto &&...TupleArg) {
    return std::array{std::forward<decltype(TupleArg)>(TupleArg)...};
  };
  return std::apply(array_from_tuple, std::forward<T>(t));
}

template <typename T, std::size_t... I>
static constexpr auto to_tuple_impl(T &&a, std::index_sequence<I...>) {
  return std::make_tuple(std::move(a[I])...);
}

template <typename T, std::size_t N>
static constexpr auto to_tuple(std::array<T, N> &&a) {
  return to_tuple_impl(std::forward<std::array<T, N>>(a),
                       std::make_index_sequence<N>{});
}

#if 0
// unrolled for replacement :(

template<int N>
struct my_algo {
template<typename C, typename B>
static std::optional<int> reverse_for_unroll(C &&CondFunc, B &&BlockFunc) {
  static_assert(N > 0);
  if (CondFunc(N))
    return BlockFunc(N);
  return{};
//  my_algo<N-1>::reverse_for_unroll(std::forward<C>(CondFunc), std::forward<B>(BlockFunc));
}
};

template<>
struct my_algo<0> {
template<typename C, typename B>
static std::optional<int> reverse_for_unroll(C &&CondFunc, B &&BlockFunc) {
  return {};
}
};
#endif

#ifdef BLOCKS_AS_ARRAYS
// rule of zero.
template <typename C, typename B, unsigned NC = 1, unsigned NB = 1>
class simd_if final {
  std::array<C, NC> Conditions;
  // check that C is a simd<>
  using CondEltTy = typename C::element_type;
  static constexpr int CondLength = C::length;
  std::array<B, NB> Blocks;

public:
  simd_if(C Condition, B Block)
      : Conditions{Condition}, Blocks{Block} {
    static_assert(NC == NB);
    // If construction. There must be 1 condition and 1 body block.
    static_assert(NC == 1);
  }
  // fix copying here.
  simd_if(std::array<C, NC> &&Conds, std::array<B, NB> &&Blks)
      : Conditions(Conds), Blocks(Blks) {
    static_assert(NB > 1);
    static_assert(NC == NB || NC == NB - 1);
  }

private:
  template<size_t I, bool Dummy>
  void exec_impl() {
    if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions[I]))
      Blocks[I]();
    else
      exec_impl<I+1, NC == NB>();
  }
  template<>
  void exec_impl<NC, false>() {
    Blocks[NC]();
  }
  template<>
  void exec_impl<NC, true>() {
  }

public:
  std::optional<int> exec() {
#if 0
    auto RetVal = my_algo<NC>::reverse_for_unroll([&](int i) {
      return __esimd_simdcf_any<CondEltTy, CondLength>(Conditions[NC - i]);}, [&](int i) { return Blocks[NC - i](); });
    if (RetVal.has_value()) // has executed return in the loop
      return RetVal;
//    for (unsigned i = 0; i < Conditions.size(); ++i)
//      if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions[i]))
//        return Blocks[i]();
    if constexpr (NC == NB)
      return {};
    else {
      static_assert(NB > 1 && NC == NB - 1);
      return Blocks.back()();
    }

#endif
    std::optional<int> RetVal = {};
#if 0
#pragma nounroll // unroll conflicts with simd cf pass. Have to investigate. Actually, it may be not unroll itself, but the loop.
    for (int i = 0; i < NC; ++i) {
      if ( __esimd_simdcf_any<CondEltTy, CondLength>(Conditions[i])) {
        RetVal = Blocks[i]();
    }
    }
    if (RetVal.has_value())
      return RetVal;

    if constexpr (NC == NB)
      return {};
    else {
      static_assert(NB > 1 && NC == NB - 1);
      return Blocks.back()();
    }
#endif
    exec_impl<0, NC == NB>();
    return {};


#if 0
    for (unsigned i = 0; i < Conditions.size(); ++i)
      if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions[i]))
        return Blocks[i]();
    if constexpr (NC == NB)
      return {};
    else {
      static_assert(NB > 1 && NC == NB - 1);
      return Blocks.back()();
    }
#endif
// Generated code is not ready for simd cf pass.
#if 0
    auto P = [](decltype(Conditions.front()) Cond) { return __esimd_simdcf_any<CondEltTy, CondLength>(Cond); };
    auto TrueCond = std::find_if(Conditions.begin(), Conditions.end(), P);
    if (TrueCond == Conditions.end()) { // else block or nothing
      if constexpr (NC == NB)
        return {};
      else {
        static_assert(NB > 1 && NC == NB - 1);
        return Blocks.back()();
      }
    }
    auto BlockPos = std::distance(Conditions.begin(), TrueCond);
    assert(BlockPos < Blocks.size());
    return Blocks[BlockPos]();
#endif
#if 0
    if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions.front()))
      return Blocks.front()();
    return {};
#endif
  }

  template <typename T, typename U,
            std::enable_if_t<std::is_same_v<std::remove_reference_t<T>, C> &&
                                 std::is_same_v<std::remove_reference_t<U>, B>,
                             bool> = true>
  simd_if<C, B, NC + 1, NB + 1> simd_elif(T &&Cond, U &&Block) {
    static_assert(NC == NB); // cannot add elif after else
    auto BlocksTup = to_tuple(std::move(Blocks));
    auto NewBlocksTup = std::tuple_cat(std::move(BlocksTup),
                                       std::make_tuple(std::forward<U>(Block)));
    std::array<B, NB + 1> NewBlocks = make_array(std::move(NewBlocksTup));

    auto ConditionsTup = to_tuple(std::move(Conditions));
    auto NewConditionsTup = std::tuple_cat(
        std::move(ConditionsTup), std::make_tuple(std::forward<T>(Cond)));
    std::array<C, NC + 1> NewConditions =
        make_array(std::move(NewConditionsTup));

    return simd_if<C, B, NC + 1, NB + 1>(std::move(NewConditions),
                                         std::move(NewBlocks));
  }
  template <typename U,
            std::enable_if_t<std::is_same_v<std::remove_reference_t<U>, B>,
                             bool> = true>
  simd_if<C, B, NC, NB + 1> simd_else(U &&Block) {
    static_assert(NB == NC); // prevent 2 else blocks.
    // have to use 'to_tuple'. Otherwise, implementation dependent UB. See
    // std::tuple_cat
    auto BlocksTup = to_tuple(std::move(Blocks));
    auto NewBlocksTup = std::tuple_cat(std::move(BlocksTup),
                                       std::make_tuple(std::forward<U>(Block)));
    std::array<B, NB + 1> NewBlocks = make_array(std::move(NewBlocksTup));
    return simd_if<C, B, NC, NB + 1>(std::move(Conditions),
                                     std::move(NewBlocks));
  }
#if 0 // type deduction is broken.
  template<typename T, typename U> // forwarding references
  static constexpr simd_if create(T &&Condition, U &&Block) {
    return simd_if(std::forward<T>(Condition), std::forward<U>(Block));
  }
#endif
};
#else

class simd_control final {
  enum class statement {
    _no_statement,
    _break,
    _continue
  };

  statement s = statement::_no_statement;

public:
  simd_control() : s(statement::_no_statement) {}
  simd_control(const simd_control &rhs) : s(rhs.s) {}
  simd_control(simd_control &&rhs) : s(rhs.s) {}
  // dirty operators
  simd_control &operator=(const simd_control &rhs) { s = rhs.s; return *this; }
  simd_control &operator=(simd_control &&rhs) { s = rhs.s; return *this; }
  simd_control& simd_break() { s = statement::_break; return *this; }
  simd_control& simd_continue() { s = statement::_continue; return *this; }
  operator bool() const { return s != statement::_break; }
};


// rule of zero.
template <typename C, unsigned NC, typename... Bs>
class _simd_if final {
  std::array<C, NC> Conditions;
  // check that C is a simd<>
  using CondEltTy = typename C::element_type;
  static constexpr int CondLength = C::length;
  std::tuple<Bs...> Blocks;
  static constexpr size_t NB = std::tuple_size<decltype(Blocks)>::value;

public:
  // fix copying here.
  _simd_if(std::array<C, NC> &&Conds, std::tuple<Bs...> &&Blks)
      : Conditions(Conds), Blocks(Blks) {
    static_assert(NB > 1);
    static_assert(NC == NB || NC == NB - 1);
  }

  _simd_if(const _simd_if &) = delete;
  _simd_if(_simd_if &&) = delete;
  _simd_if &operator=(const _simd_if &) = delete;
  _simd_if &operator=(_simd_if &&) = delete;

private:
  // Not a good idea. Private template method
  template<size_t I, bool Dummy>
  void exec_impl() {
    if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions[I]))
      std::get<I>(Blocks)();
    else
      exec_impl<I+1, NC == NB>();
  }
  template<>
  void exec_impl<NC, false>() {
    std::get<NC>(Blocks)();
  }
  template<>
  void exec_impl<NC, true>() {
  }

public:
  void exec() {
#if 0
    auto RetVal = my_algo<NC>::reverse_for_unroll([&](int i) {
      return __esimd_simdcf_any<CondEltTy, CondLength>(Conditions[NC - i]);}, [&](int i) { return Blocks[NC - i](); });
    if (RetVal.has_value()) // has executed return in the loop
      return RetVal;
//    for (unsigned i = 0; i < Conditions.size(); ++i)
//      if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions[i]))
//        return Blocks[i]();
    if constexpr (NC == NB)
      return {};
    else {
      static_assert(NB > 1 && NC == NB - 1);
      return Blocks.back()();
    }

#endif
#if 0
#pragma nounroll // unroll conflicts with simd cf pass. Have to investigate. Actually, it may be not unroll itself, but the loop.
    for (int i = 0; i < NC; ++i) {
      if ( __esimd_simdcf_any<CondEltTy, CondLength>(Conditions[i])) {
        RetVal = Blocks[i]();
    }
    }
    if (RetVal.has_value())
      return RetVal;

    if constexpr (NC == NB)
      return {};
    else {
      static_assert(NB > 1 && NC == NB - 1);
      return Blocks.back()();
    }
#endif
    exec_impl<0, true>();
    return;


#if 0
    for (unsigned i = 0; i < Conditions.size(); ++i)
      if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions[i]))
        return Blocks[i]();
    if constexpr (NC == NB)
      return {};
    else {
      static_assert(NB > 1 && NC == NB - 1);
      return Blocks.back()();
    }
#endif
// Generated code is not ready for simd cf pass.
#if 0
    auto P = [](decltype(Conditions.front()) Cond) { return __esimd_simdcf_any<CondEltTy, CondLength>(Cond); };
    auto TrueCond = std::find_if(Conditions.begin(), Conditions.end(), P);
    if (TrueCond == Conditions.end()) { // else block or nothing
      if constexpr (NC == NB)
        return {};
      else {
        static_assert(NB > 1 && NC == NB - 1);
        return Blocks.back()();
      }
    }
    auto BlockPos = std::distance(Conditions.begin(), TrueCond);
    assert(BlockPos < Blocks.size());
    return Blocks[BlockPos]();
#endif
#if 0
    if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions.front()))
      return Blocks.front()();
    return {};
#endif
  }

  // ONLY FOR RVALUES
  template <typename T, typename U,
            std::enable_if_t<std::is_same_v<std::remove_reference_t<T>, C>,
                             bool> = true>
  _simd_if<C, NC + 1, Bs..., U> simd_elif(T Cond, U Block) {
    static_assert(NC == NB); // cannot add elif after else
    auto NewBlocks = std::tuple_cat(std::move(Blocks),
                                       std::make_tuple(Block));

    auto ConditionsTup = to_tuple(std::move(Conditions));
    auto NewConditionsTup = std::tuple_cat(
        std::move(ConditionsTup), std::make_tuple(std::forward<T>(Cond)));
    std::array<C, NC + 1> NewConditions =
        make_array(std::move(NewConditionsTup));

    return _simd_if<C, NC + 1, Bs..., U>(std::move(NewConditions),
                                         std::move(NewBlocks));
  }
  template <typename U>
  _simd_if<C, NC, Bs..., U> simd_else(U Block) {
    static_assert(NB == NC); // prevent 2 else blocks.
    auto NewBlocks = std::tuple_cat(std::move(Blocks),
                                       std::make_tuple(Block));
    return _simd_if<C, NC, Bs..., U>(std::move(Conditions),
                                     std::move(NewBlocks));
  }
#if 0 // type deduction is broken.
  template<typename T, typename U> // forwarding references
  static constexpr simd_if create(T &&Condition, U &&Block) {
    return simd_if(std::forward<T>(Condition), std::forward<U>(Block));
  }
#endif
};

template <typename C, typename B>
class simd_if final {
  C Conditions;
  // check that C is a simd<>
  using CondEltTy = typename C::element_type;
  static constexpr int CondLength = C::length;
  B Blocks;

public:
  template<std::enable_if_t<std::is_invocable_v<B>, bool> = true>
  simd_if(C Condition, B Block)
      : Conditions{Condition}, Blocks{Block} {
  }

  simd_if(const simd_if &) = delete;
  simd_if(simd_if &&) = delete;
  simd_if &operator=(const simd_if &) = delete;
  simd_if &operator=(simd_if &&) = delete;

private:
  std::optional<bool> exec_impl() {
    std::optional<bool> RetVal = {};
    if (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions))
      RetVal = Blocks();
    return RetVal;
  }

public:
  std::optional<bool> exec() {
    return exec_impl();
  }

  template <typename T, typename U,
            std::enable_if_t<std::is_same_v<std::remove_reference_t<T>, C>, bool> = true>
  _simd_if<C, 2, B, U> simd_elif(T Cond, U Block) {
    // forward
    std::array<C, 2> NewConditions = {{Conditions, Cond}};
    std::tuple<B, U> NewBlocks = std::make_tuple(Blocks, Block);
    return _simd_if<C, 2, B, U>(std::move(NewConditions), std::move(NewBlocks));
  }
  template <typename U>
  _simd_if<C, 1, B, U> simd_else(U Block) {
    std::array<C, 1> NewConditions = {{Conditions}};
    std::tuple<B, U> NewBlocks = std::make_tuple(Blocks, Block);
    return _simd_if<C, 1, B, U>(std::move(NewConditions), std::move(NewBlocks));
  }
};

template <typename C, typename B>
class simd_while final {
  C Conditions;
  // check that C is a simd<>
  using CondEltTy = typename decltype(Conditions())::element_type;
  static constexpr int CondLength = decltype(Conditions())::length;
  B Blocks;

public:
  // and for C
  template<std::enable_if_t<std::is_invocable_v<B>, bool> = true>
  simd_while(C Condition, B Block)
      : Conditions{Condition}, Blocks{Block} {
  }

  simd_while(const simd_while &) = delete;
  simd_while(simd_while &&) = delete;
  simd_while &operator=(const simd_while &) = delete;
  simd_while &operator=(simd_while &&) = delete;

public:
  void exec() {
    if constexpr (std::is_same_v<decltype(Blocks()), void>) {
      while (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions()))
        Blocks();
    } else {
      // only simd_control
      simd_control control;
      while (__esimd_simdcf_any<CondEltTy, CondLength>(Conditions()) && control)
        control = Blocks();
    }
  }

};


#endif

} // namespace cls

} // namespace gpu
} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#ifndef __SYCL_DEVICE_ONLY__
template <typename Ty, int N>
std::ostream &operator<<(std::ostream &OS,
                         const sycl::INTEL::gpu::simd<Ty, N> &V) {
  OS << "{";
  for (int I = 0; I < N; I++) {
    OS << V[I];
    if (I < N - 1)
      OS << ",";
  }
  OS << "}";
  return OS;
}
#endif
