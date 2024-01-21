/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <list>
#include <type_traits>
#include <vector>

#include <mcl/assert.hpp>

#include "dynarmic/backend/arm64/a64_address_space.h"
#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/always_false.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/variant_util.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend::Arm64 {

namespace {
void AppendNextBlocksToList(std::list<IR::LocationDescriptor>& next, const IR::Terminal& terminal) {
    Common::VisitVariant<void>(terminal, [&](auto x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, IR::Term::Invalid>) {
            ASSERT_FALSE("Invalid terminal");
        } else if constexpr (std::is_same_v<T, IR::Term::ReturnToDispatch>) {
            // Nothing
        } else if constexpr (std::is_same_v<T, IR::Term::LinkBlock>) {
            next.push_back(x.next);
        } else if constexpr (std::is_same_v<T, IR::Term::LinkBlockFast>) {
            next.push_back(x.next);
        } else if constexpr (std::is_same_v<T, IR::Term::PopRSBHint>) {
            // Nothing
        } else if constexpr (std::is_same_v<T, IR::Term::FastDispatchHint>) {
            // Nothing
        } else if constexpr (std::is_same_v<T, IR::Term::If>) {
            AppendNextBlocksToList(next, x.then_);
            AppendNextBlocksToList(next, x.else_);
        } else if constexpr (std::is_same_v<T, IR::Term::CheckBit>) {
            AppendNextBlocksToList(next, x.then_);
            AppendNextBlocksToList(next, x.else_);
        } else if constexpr (std::is_same_v<T, IR::Term::CheckHalt>) {
            AppendNextBlocksToList(next, x.else_);
        } else {
            static_assert(Common::always_false_v<T>);
        }
    });
}
}  // namespace

AddressSpace::AddressSpace(EmitConfig emit_config, size_t code_cache_size)
        : emit_config(emit_config)
        , code_cache_size(code_cache_size)
        , mem(code_cache_size)
        , code(mem.ptr())
        , fastmem_manager(exception_handler) {
    ASSERT_MSG(code_cache_size <= 128 * 1024 * 1024, "code_cache_size > 128 MiB not currently supported");

    exception_handler.Register(mem, code_cache_size);
    exception_handler.SetFastmemCallback([this](u64 host_pc) {
        return FastmemCallback(host_pc);
    });
}

AddressSpace::~AddressSpace() = default;

CodePtr AddressSpace::Get(IR::LocationDescriptor descriptor) {
    if (const auto iter = block_entries.find(descriptor); iter != block_entries.end()) {
        return iter->second;
    }
    return nullptr;
}

std::optional<IR::LocationDescriptor> AddressSpace::ReverseGetLocation(CodePtr host_pc) {
    if (auto iter = reverse_block_entries.upper_bound(host_pc); iter != reverse_block_entries.begin()) {
        // upper_bound locates the first value greater than host_pc, so we need to decrement
        --iter;
        return iter->second;
    }
    return std::nullopt;
}

CodePtr AddressSpace::ReverseGetEntryPoint(CodePtr host_pc) {
    if (auto iter = reverse_block_entries.upper_bound(host_pc); iter != reverse_block_entries.begin()) {
        // upper_bound locates the first value greater than host_pc, so we need to decrement
        --iter;
        return iter->first;
    }
    return nullptr;
}

CodePtr AddressSpace::GetOrEmit(IR::LocationDescriptor descriptor) {
    if (CodePtr block_entry = Get(descriptor)) {
        return block_entry;
    }

    if (IsNearlyFull()) {
        ClearCache();
    }

    return Compile(descriptor);
}

void AddressSpace::InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& descriptors) {
    mem.unprotect();

    for (const auto& descriptor : descriptors) {
        const auto iter = block_entries.find(descriptor);
        if (iter == block_entries.end()) {
            continue;
        }

        // Unlink before removal because InvalidateBasicBlocks can be called within a fastmem callback,
        // and the currently executing block may have references to itself which need to be unlinked.
        RelinkForDescriptor(descriptor, nullptr);

        block_entries.erase(iter);
    }

    mem.protect();
}

void AddressSpace::ClearCache() {
    block_entries.clear();
    reverse_block_entries.clear();
    block_infos.clear();
    block_references.clear();
    code.set_ptr(prelude_info.end_of_prelude);
}

size_t AddressSpace::GetRemainingSize() {
    return code_cache_size - (code.ptr<CodePtr>() - reinterpret_cast<CodePtr>(mem.ptr()));
}

bool AddressSpace::IsNearlyFull() {
    return GetRemainingSize() < 1024 * 1024;
}

CodePtr AddressSpace::Compile(IR::LocationDescriptor descriptor) {
    const CodePtr starting_position = code.ptr<CodePtr>();

    mem.unprotect();

    std::list<IR::LocationDescriptor> next;

    const auto do_block = [this, &next](IR::LocationDescriptor descriptor) {
        IR::Block ir_block = GenerateIR(descriptor);
        AppendNextBlocksToList(next, ir_block.GetTerminal());
        const EmittedBlockInfo block_info = Emit(std::move(ir_block));
        return block_info.entry_point;
    };

    CodePtr result = do_block(descriptor);
    if (emit_config.HasOptimization(OptimizationFlag::MultiBlockCompilation)) {
        while (!next.empty() && !IsNearlyFull()) {
            const IR::LocationDescriptor n = next.front();
            next.pop_front();
            if (!Get(n))
                do_block(n);
        }
    }

    mem.invalidate(reinterpret_cast<u32*>(starting_position), static_cast<size_t>(code.ptr<CodePtr>() - starting_position));
    mem.protect();

    return result;
}

EmittedBlockInfo AddressSpace::Emit(IR::Block block) {
    EmittedBlockInfo block_info = EmitArm64(code, std::move(block), emit_config, fastmem_manager);

    ASSERT(block_entries.emplace(block.Location(), block_info.entry_point).second);
    ASSERT(reverse_block_entries.emplace(block_info.entry_point, block.Location()).second);
    ASSERT(block_infos.emplace(block_info.entry_point, block_info).second);

    Link(block_info);
    RelinkForDescriptor(block.Location(), block_info.entry_point);

    RegisterNewBasicBlock(block, block_info);

    return block_info;
}

static void LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list, void* return_to_dispatcher) {
    using namespace oaknut;
    using namespace oaknut::util;

    for (auto [ptr_offset, type] : block_relocations_list) {
        CodeGenerator c{reinterpret_cast<u32*>(entry_point + ptr_offset)};

        switch (type) {
        case BlockRelocationType::Branch:
            if (target_ptr) {
                c.B((void*)target_ptr);
            } else {
                c.NOP();
            }
            break;
        case BlockRelocationType::MoveToScratch1:
            if (target_ptr) {
                c.ADRL(Xscratch1, (void*)target_ptr);
            } else {
                c.ADRL(Xscratch1, return_to_dispatcher);
            }
            break;
        default:
            ASSERT_FALSE("Invalid BlockRelocationType");
        }
    }
}

void AddressSpace::Link(EmittedBlockInfo& block_info) {
    using namespace oaknut;
    using namespace oaknut::util;

    for (auto [ptr_offset, target] : block_info.relocations) {
        CodeGenerator c{reinterpret_cast<u32*>(block_info.entry_point + ptr_offset)};

        switch (target) {
        case LinkTarget::ReturnToDispatcher:
            c.B(prelude_info.return_to_dispatcher);
            break;
        case LinkTarget::ReturnFromRunCode:
            c.B(prelude_info.return_from_run_code);
            break;
        case LinkTarget::ReadMemory8:
            c.BL(prelude_info.read_memory_8);
            break;
        case LinkTarget::ReadMemory16:
            c.BL(prelude_info.read_memory_16);
            break;
        case LinkTarget::ReadMemory32:
            c.BL(prelude_info.read_memory_32);
            break;
        case LinkTarget::ReadMemory64:
            c.BL(prelude_info.read_memory_64);
            break;
        case LinkTarget::ReadMemory128:
            c.BL(prelude_info.read_memory_128);
            break;
        case LinkTarget::WrappedReadMemory8:
            c.BL(prelude_info.wrapped_read_memory_8);
            break;
        case LinkTarget::WrappedReadMemory16:
            c.BL(prelude_info.wrapped_read_memory_16);
            break;
        case LinkTarget::WrappedReadMemory32:
            c.BL(prelude_info.wrapped_read_memory_32);
            break;
        case LinkTarget::WrappedReadMemory64:
            c.BL(prelude_info.wrapped_read_memory_64);
            break;
        case LinkTarget::WrappedReadMemory128:
            c.BL(prelude_info.wrapped_read_memory_128);
            break;
        case LinkTarget::ExclusiveReadMemory8:
            c.BL(prelude_info.exclusive_read_memory_8);
            break;
        case LinkTarget::ExclusiveReadMemory16:
            c.BL(prelude_info.exclusive_read_memory_16);
            break;
        case LinkTarget::ExclusiveReadMemory32:
            c.BL(prelude_info.exclusive_read_memory_32);
            break;
        case LinkTarget::ExclusiveReadMemory64:
            c.BL(prelude_info.exclusive_read_memory_64);
            break;
        case LinkTarget::ExclusiveReadMemory128:
            c.BL(prelude_info.exclusive_read_memory_128);
            break;
        case LinkTarget::WriteMemory8:
            c.BL(prelude_info.write_memory_8);
            break;
        case LinkTarget::WriteMemory16:
            c.BL(prelude_info.write_memory_16);
            break;
        case LinkTarget::WriteMemory32:
            c.BL(prelude_info.write_memory_32);
            break;
        case LinkTarget::WriteMemory64:
            c.BL(prelude_info.write_memory_64);
            break;
        case LinkTarget::WriteMemory128:
            c.BL(prelude_info.write_memory_128);
            break;
        case LinkTarget::WrappedWriteMemory8:
            c.BL(prelude_info.wrapped_write_memory_8);
            break;
        case LinkTarget::WrappedWriteMemory16:
            c.BL(prelude_info.wrapped_write_memory_16);
            break;
        case LinkTarget::WrappedWriteMemory32:
            c.BL(prelude_info.wrapped_write_memory_32);
            break;
        case LinkTarget::WrappedWriteMemory64:
            c.BL(prelude_info.wrapped_write_memory_64);
            break;
        case LinkTarget::WrappedWriteMemory128:
            c.BL(prelude_info.wrapped_write_memory_128);
            break;
        case LinkTarget::ExclusiveWriteMemory8:
            c.BL(prelude_info.exclusive_write_memory_8);
            break;
        case LinkTarget::ExclusiveWriteMemory16:
            c.BL(prelude_info.exclusive_write_memory_16);
            break;
        case LinkTarget::ExclusiveWriteMemory32:
            c.BL(prelude_info.exclusive_write_memory_32);
            break;
        case LinkTarget::ExclusiveWriteMemory64:
            c.BL(prelude_info.exclusive_write_memory_64);
            break;
        case LinkTarget::ExclusiveWriteMemory128:
            c.BL(prelude_info.exclusive_write_memory_128);
            break;
        case LinkTarget::CallSVC:
            c.BL(prelude_info.call_svc);
            break;
        case LinkTarget::ExceptionRaised:
            c.BL(prelude_info.exception_raised);
            break;
        case LinkTarget::InstructionSynchronizationBarrierRaised:
            c.BL(prelude_info.isb_raised);
            break;
        case LinkTarget::InstructionCacheOperationRaised:
            c.BL(prelude_info.ic_raised);
            break;
        case LinkTarget::DataCacheOperationRaised:
            c.BL(prelude_info.dc_raised);
            break;
        case LinkTarget::GetCNTPCT:
            c.BL(prelude_info.get_cntpct);
            break;
        case LinkTarget::AddTicks:
            c.BL(prelude_info.add_ticks);
            break;
        case LinkTarget::GetTicksRemaining:
            c.BL(prelude_info.get_ticks_remaining);
            break;
        default:
            ASSERT_FALSE("Invalid relocation target");
        }
    }

    for (auto [target_descriptor, list] : block_info.block_relocations) {
        block_references[target_descriptor].emplace(block_info.entry_point);
        LinkBlockLinks(block_info.entry_point, Get(target_descriptor), list, prelude_info.return_to_dispatcher);
    }
}

void AddressSpace::RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr) {
    for (auto code_ptr : block_references[target_descriptor]) {
        if (auto block_iter = block_infos.find(code_ptr); block_iter != block_infos.end()) {
            const EmittedBlockInfo& block_info = block_iter->second;

            if (auto relocation_iter = block_info.block_relocations.find(target_descriptor); relocation_iter != block_info.block_relocations.end()) {
                LinkBlockLinks(block_info.entry_point, target_ptr, relocation_iter->second, prelude_info.return_to_dispatcher);
            }

            mem.invalidate(reinterpret_cast<u32*>(block_info.entry_point), block_info.size);
        }
    }
}

FakeCall AddressSpace::FastmemCallback(u64 host_pc) {
    {
        const auto host_ptr = mcl::bit_cast<CodePtr>(host_pc);

        const auto entry_point = ReverseGetEntryPoint(host_ptr);
        if (!entry_point) {
            goto fail;
        }

        const auto block_info = block_infos.find(entry_point);
        if (block_info == block_infos.end()) {
            goto fail;
        }

        const auto patch_entry = block_info->second.fastmem_patch_info.find(host_ptr - entry_point);
        if (patch_entry == block_info->second.fastmem_patch_info.end()) {
            goto fail;
        }

        const auto fc = patch_entry->second.fc;

        if (patch_entry->second.recompile) {
            const auto marker = patch_entry->second.marker;
            fastmem_manager.MarkDoNotFastmem(marker);
            InvalidateBasicBlocks({std::get<0>(marker)});
        }

        return fc;
    }

fail:
    fmt::print("dynarmic: Segfault happened within JITted code at host_pc = {:016x}\n", host_pc);
    fmt::print("Segfault wasn't at a fastmem patch location!\n");
    ASSERT_FALSE("segfault");
}

}  // namespace Dynarmic::Backend::Arm64
