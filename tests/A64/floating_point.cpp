/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <catch2/catch.hpp>

#include "./testenv.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/op/FPRecipEstimate.h"
#include "dynarmic/interface/A64/a64.h"

using namespace Dynarmic;

const std::initializer_list<u32> special_values_32{
    // Special values
    0x0000'0000,  // positive zero
    0x0000'0001,  // smallest positive denormal
    0x0000'1000,
    0x007F'FFFF,  // largest positive denormal
    0x0080'0000,  // smallest positive normal
    0x0080'0002,
    0x3F80'0000,  // 1.0
    0x7F7F'FFFF,  // largest positive normal
    0x7F80'0000,  // positive infinity
    0x7F80'0001,  // first positive SNaN
    0x7FBF'FFFF,  // last positive SNaN
    0x7FC0'0000,  // first positive QNaN
    0x7FFF'FFFF,  // last positive QNaN
    0x8000'0000,  // negative zero
    0x8000'0001,  // smallest negative denormal
    0x8000'1000,
    0x807F'FFFF,  // largest negative denormal
    0x8080'0000,  // smallest negative normal
    0x8080'0002,
    0xBFF0'0000,  // -1.0
    0xFF7F'FFFF,  // largest negative normal
    0xFF80'0000,  // negative infinity
    0xFF80'0001,  // first negative SNaN
    0xFFBF'FFFF,  // last negative SNaN
    0xFFC0'0000,  // first negative QNaN
    0xFFFF'FFFF,  // last negative QNaN

    0x7E00'0000,  // 2^125
    0x7E80'0000,  // 2^126
    0xC7C0'0000,  // -2^125
    0xC7D0'0000,  // -2^126

    // Some typical numbers
    0x3FC0'0000,  // 1.5
    0x447A'0000,  // 1000
    0xC040'0000,  // -3
};

const std::initializer_list<u64> special_values_64{
    // Special values
    0x0000'0000'0000'0000,  // positive zero
    0x0000'0000'0000'0001,  // smallest positive denormal
    0x0000'0000'0100'0000,
    0x000F'FFFF'FFFF'FFFF,  // largest positive denormal
    0x0010'0000'0000'0000,  // smallest positive normal
    0x0010'0000'0000'0002,
    0x3FF0'0000'0000'0000,  // 1.0
    0x7FEF'FFFF'FFFF'FFFF,  // largest positive normal
    0x7FF0'0000'0000'0000,  // positive infinity
    0x7FF0'0000'0000'0001,  // first positive SNaN
    0x7FF7'FFFF'FFFF'FFFF,  // last positive SNaN
    0x7FF8'0000'0000'0000,  // first positive QNaN
    0x7FFF'FFFF'FFFF'FFFF,  // last positive QNaN
    0x8000'0000'0000'0000,  // negative zero
    0x8000'0000'0000'0001,  // smallest negative denormal
    0x8000'0000'0100'0000,
    0x800F'FFFF'FFFF'FFFF,  // largest negative denormal
    0x8010'0000'0000'0000,  // smallest negative normal
    0x8010'0000'0000'0002,
    0xBFF0'0000'0000'0000,  // -1.0
    0xFFEF'FFFF'FFFF'FFFF,  // largest negative normal
    0xFFF0'0000'0000'0000,  // negative infinity
    0xFFF0'0000'0000'0001,  // first negative SNaN
    0xFFF7'FFFF'FFFF'FFFF,  // last negative SNaN
    0xFFF8'0000'0000'0000,  // first negative QNaN
    0xFFFF'FFFF'FFFF'FFFF,  // last negative QNaN

    0x3800'0000'0000'0000,  // 2^(-127)
    0x3810'0000'0000'0000,  // 2^(-126)
    0xB800'0000'0000'0000,  // -2^(-127)
    0xB810'0000'0000'0000,  // -2^(-126)
    0x3800'1234'5678'9ABC, 0x3810'1234'5678'9ABC, 0xB800'1234'5678'9ABC, 0xB810'1234'5678'9ABC,

    0x3680'0000'0000'0000,  // 2^(-150)
    0x36A0'0000'0000'0000,  // 2^(-149)
    0x36B0'0000'0000'0000,  // 2^(-148)
    0xB680'0000'0000'0000,  // -2^(-150)
    0xB6A0'0000'0000'0000,  // -2^(-149)
    0xB6B0'0000'0000'0000,  // -2^(-148)
    0x3680'1234'5678'9ABC, 0x36A0'1234'5678'9ABC, 0x36B0'1234'5678'9ABC, 0xB680'1234'5678'9ABC,
    0xB6A0'1234'5678'9ABC, 0xB6B0'1234'5678'9ABC,

    0x47C0'0000'0000'0000,  // 2^125
    0x47D0'0000'0000'0000,  // 2^126
    0xC7C0'0000'0000'0000,  // -2^125
    0xC7D0'0000'0000'0000,  // -2^126

    0x37F0'0000'0000'0000,  // 2^(-128)
    0x37E0'0000'0000'0000,  // 2^(-129)
    0xB7F0'0000'0000'0000,  // -2^(-128)
    0xB7E0'0000'0000'0000,  // -2^(-129)

    // Some typical numbers
    0x3FF8'0000'0000'0000,  // 1.5
    0x408F'4000'0000'0000,  // 1000
    0xC008'0000'0000'0000,  // -3
};

TEST_CASE("FRECPE (32-bit)", "[a64]") {
    A64TestEnv env;

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    env.code_mem_start_address = 100;

    env.code_mem.emplace_back(0x1e270000);  // FMOV S0, W0
    env.code_mem.emplace_back(0x5ea1d800);  // FRECPE S0, S0
    env.code_mem.emplace_back(0x1e260000);  // FMOV W0, S0
    env.code_mem.emplace_back(0x14000000);  // B .

    const auto run = [&](u32 x) {
        jit.SetRegister(0, x);

        jit.SetPC(100);
        env.ticks_left = 4;
        jit.Run();

        return static_cast<u32>(jit.GetRegister(0));
    };

    FP::FPCR fpcr{};
    FP::FPSR fpsr{};

    for (const u32 special_value : special_values_32) {
        INFO(special_value);
        REQUIRE(run(special_value) == FP::FPRecipEstimate<u32>(special_value, fpcr, fpsr));
    }

    for (u64 i = 0; i < 0x100000000; i += 0x7F) {
        const u32 value = static_cast<u32>(i);
        INFO(value);
        REQUIRE(run(value) == FP::FPRecipEstimate<u32>(value, fpcr, fpsr));
    }
}

TEST_CASE("FRECPE (64-bit)", "[a64]") {
    A64TestEnv env;

    A64::UserConfig conf{&env};
    A64::Jit jit{conf};

    env.code_mem_start_address = 100;

    env.code_mem.emplace_back(0x9e670000);  // FMOV D0, X0
    env.code_mem.emplace_back(0x5ee1d800);  // FRECPE D0, D0
    env.code_mem.emplace_back(0x9e660000);  // FMOV X0, D0
    env.code_mem.emplace_back(0x14000000);  // B .

    const auto run = [&](u64 x) {
        jit.SetRegister(0, x);

        jit.SetPC(100);
        env.ticks_left = 4;
        jit.Run();

        return jit.GetRegister(0);
    };

    FP::FPCR fpcr{};
    FP::FPSR fpsr{};

    for (const u64 special_value : special_values_64) {
        INFO(special_value);
        REQUIRE(run(special_value) == FP::FPRecipEstimate<u64>(special_value, fpcr, fpsr));
    }

    for (u64 i = 0; i < 0xFFF0'0000'0000'0000; i += ((1ull << 39) - 1)) {
        INFO(i);
        REQUIRE(run(i) == FP::FPRecipEstimate<u64>(i, fpcr, fpsr));
    }
}
