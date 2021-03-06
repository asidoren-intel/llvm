//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// unordered_map(unordered_map&& u);

// UNSUPPORTED: c++03
// UNSUPPORTED: libcxx-no-debug-mode

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DEBUG=1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <unordered_map>
#include <cassert>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
    std::unordered_map<int, int> s1 = {{1, 1}, {2, 2}, {3, 3}};
    std::unordered_map<int, int>::iterator i = s1.begin();
    std::pair<const int, int> k = *i;
    std::unordered_map<int, int> s2 = std::move(s1);
    assert(*i == k);
    s2.erase(i);
    assert(s2.size() == 2);

    return 0;
}
