// RUN: %clang_cc1 -fsycl -fsycl-is-host -Wno-sycl-2017-compat -fsyntax-only -verify %s
// expected-no-diagnostics

[[intel::reqd_work_group_size(4)]] void f4x1x1() {}

[[intel::reqd_work_group_size(16)]] void f16x1x1() {}

[[intel::reqd_work_group_size(32, 32, 32)]] void f32x32x32() {}

class Functor64 {
public:
  [[intel::reqd_work_group_size(64, 64, 64)]] void operator()() const {}
};
