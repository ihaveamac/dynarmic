/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <catch2/catch_test_macros.hpp>
#include <mcl/stdint.hpp>

#include "./testenv.h"
#include "dynarmic/interface/A32/a32.h"
#include "dynarmic/interface/exclusive_monitor.h"

static Dynarmic::ExclusiveMonitor exclusive_monitor{1};

static Dynarmic::A32::UserConfig GetUserConfig(Dynarmic::A32::UserCallbacks* testenv) {
    Dynarmic::A32::UserConfig user_config;
    user_config.callbacks = testenv;
    user_config.global_monitor = &exclusive_monitor;
    return user_config;
}

TEST_CASE("thumb: try all instructions", "[thumb][A32][.]") {
    ThumbTestEnv test_env;
    Dynarmic::A32::Jit jit{GetUserConfig(&test_env)};

    for (u64 inst = 0; inst < 0x1'0000'0000; inst++) {
        test_env.do_assert = false;
        test_env.code_mem.clear();
        test_env.code_mem.emplace_back(static_cast<u16>(inst));
        test_env.code_mem.emplace_back(static_cast<u16>(inst >> 16));
        jit.SetCpsr(0);  // Not Thumb
        jit.SetFpscr(0);
        jit.Regs()[15] = 0;
        test_env.ticks_left = 1;
        jit.ClearCache();
        jit.Run();
        if (inst % 1000 == 0)
            std::printf("%08llx\r", inst);
    }
}

TEST_CASE("arm: try all instructions", "[arm][A32][.]") {
    ArmTestEnv test_env;
    Dynarmic::A32::Jit jit{GetUserConfig(&test_env)};

    for (u64 inst = 0xe180'0000; inst < 0x1'0000'0000; inst++) {
        test_env.do_assert = false;
        test_env.code_mem.clear();
        test_env.code_mem.emplace_back(static_cast<u32>(inst));
        jit.SetCpsr(0);  // Not Thumb
        jit.SetFpscr(0);
        jit.Regs()[15] = 0;
        test_env.ticks_left = 1;
        jit.ClearCache();
        jit.Run();
        if (inst % 1000 == 0)
            std::printf("%08llx\r", inst);
    }
}
