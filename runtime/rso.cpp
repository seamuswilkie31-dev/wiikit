// wiikit runtime — RSO modules, the SDK's relocatable code.
//
// A game built with the SDK's RSO library loads part of its code at run time
// (.rso files, read into its heap and linked by name against the executable).
// The recompiler links each module at a virtual address of its own and
// recompiles it with the executable (wiikit/rso.py): the generated code reads
// its relocated immediates from the module the game loaded (g_rso_base), and
// this file turns the run-time addresses of a module's code (its virtual
// methods, its callbacks, a switch's cases, its _prolog) into the virtual
// ones the dispatch table knows. It learns where each module is from the
// SDK's own linker: RSOLinkListFixed (RSOLinkList's body) and RSOUnLinkList.
#include "rt.h"
#include <cstdio>
#include <string>

extern const RsoModule g_rso_modules[];
extern const size_t g_rso_nmodules;
extern uint32_t g_rso_base[];

namespace {

PPCFunc orig_link = nullptr, orig_unlink = nullptr;

// After RSOLocateObject, the header holds addresses: its name at +0x10, the
// name's length at +0x14.
std::string module_name(uint32_t rso) {
    uint32_t p = ld32(rso + 0x10), n = ld32(rso + 0x14);
    std::string s;
    for (uint32_t i = 0; i < n && i < 256; ++i) s += (char)ld8(p + i);
    return s;
}

void hle_RSOLinkListFixed(PPCContext& c) {
    uint32_t rso = c.r[3];
    orig_link(c);
    if (!c.r[3]) return;
    std::string name = module_name(rso);
    for (size_t k = 0; k < g_rso_nmodules; ++k) {
        if (name == g_rso_modules[k].name) {
            g_rso_base[k] = rso;
            std::fprintf(stderr, "rso: %s linked at %08X (virtual %08X)\n", name.c_str(), rso,
                         g_rso_modules[k].vbase);
            return;
        }
    }
    std::fprintf(stderr, "rso: %s linked at %08X, not recompiled: its code cannot run\n", name.c_str(), rso);
}

void hle_RSOUnLinkList(PPCContext& c) {
    uint32_t rso = c.r[3];
    orig_unlink(c);
    for (size_t k = 0; k < g_rso_nmodules; ++k) {
        if (g_rso_base[k] == rso) {
            std::fprintf(stderr, "rso: %s unlinked\n", g_rso_modules[k].name);
            g_rso_base[k] = 0;
        }
    }
}

}  // namespace

uint32_t rso_virtual(uint32_t addr) {
    for (size_t k = 0; k < g_rso_nmodules; ++k) {
        uint32_t b = g_rso_base[k];
        if (b && addr - b < g_rso_modules[k].size) return g_rso_modules[k].vbase + (addr - b);
    }
    return 0;
}

const RsoModule* rso_module_at(uint32_t vaddr) {
    for (size_t k = 0; k < g_rso_nmodules; ++k) {
        const RsoModule& m = g_rso_modules[k];
        if (vaddr - m.vbase < m.size) return &m;
    }
    return nullptr;
}

void rso_install() {
    if (!g_rso_nmodules) return;
    orig_link = ppc_hook("RSOLinkListFixed", hle_RSOLinkListFixed);
    orig_unlink = ppc_hook("RSOUnLinkList", hle_RSOUnLinkList);
    if (!orig_link || !orig_unlink) rt_die("RSO modules are recompiled but RSOLinkListFixed or RSOUnLinkList is not named");
}
