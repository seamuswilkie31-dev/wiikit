"""Recompile a symbolised Wii executable to C++.

    python -m wiikit.recomp GAME.elf --out build/recomp [--per-file 8000] [--no-comments]
                                     [--hooks extra-hooks.txt]
                                     [--sel STATIC.sel --rso MODULE.rso ...]

Writes, into --out:
    funcs.h          prototypes of every recompiled entry point
    recomp_NNN.cpp   the functions, split by instruction count
    table.cpp        the sorted address -> function table for indirect calls
    hooks.cpp        the replaceable functions (runtime/hooks.txt and --hooks):
                     a slot the runtime can fill, and the recompiled original
    CMakeLists.txt   the library, the link check, and the runtime targets
    report.txt       units, entries, switch tables, anything not handled
    symbols.tsv      the units, named, in wiikit.dol's --symbols format (for
                     wiiboot --symbols: a stripped executable's names in logs)
    rso.cpp          the RSO modules recompiled with the executable (--rso):
                     their names and virtual addresses, for the runtime

Only symbolised images for now (ELF with .symtab): the symbols give function
boundaries and the hook names. The generated code needs wiikit/runtime/ppc.h.
"""
import argparse
import os
import time

from .. import cw
from ..dol import Image
from . import emit as E
from .program import Program

RUNTIME = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "runtime"))


class Out:
    """A generated file, written only if its content changed: a rebuild after
    a small change (a new hook) recompiles only what it touched."""

    def __init__(self, path):
        self.path, self.parts = path, []

    def write(self, s):
        self.parts.append(s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        text = "".join(self.parts)
        try:
            with open(self.path, encoding="utf-8", newline="") as f:
                if f.read() == text:
                    return
        except OSError:
            pass
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            f.write(text)


class FnCtx:
    """Control-flow answers for one entry's body."""

    def __init__(self, prog, entry, entries, module=None):
        self.prog, self.entry, self.entries = prog, entry, entries
        self.unit, _ = prog.body(entry)
        self.tables = prog.tables
        self.module = module              # the RSO module (wiikit.rso.Linked) this code is in

    # In a module, the code is linked at a virtual address and runs wherever
    # the game loaded it (wiikit/rso.py): relocated immediates are read from
    # the loaded code, and a run-time code address is made virtual.
    def relocated_half(self, a):
        m = self.module
        if m is None or a not in m.imm_sites:
            return None
        return f"ld16(g_rso_base[{m.index}] + 0x{a + 2 - m.vbase:X}u)"

    def code_address(self, expr):
        m = self.module
        if m is None:
            return expr
        return f"({expr} - g_rso_base[{m.index}] + 0x{m.vbase:08X}u)"

    # A loop that only reads the small-data area (r13, r2: never hardware)
    # and compares can end only when an interrupt handler changes memory, since
    # one guest thread runs at a time: the SDK's idle loop in SelectThread,
    # spinning on RunQueueBits. Its back-edge waits for the interrupt line.
    IDLE_OPS = {"lwz", "lhz", "lha", "lbz", "cmpi", "cmpli", "cmp", "cmpl", "rlwinm", "ori", "bc"}

    def idle_loop(self, t, src):
        if not hasattr(self, "_ins"):
            self._ins = dict(self.unit.ins)
        body = [self._ins.get(a) for a in range(t, src + 4, 4)]
        if not body or len(body) > 8 or any(i is None or i.op not in self.IDLE_OPS for i in body):
            return False
        if body[-1].op != "bc" or any(i.op == "bc" for i in body[:-1]):
            return False
        if any(i.op == "ori" and (i.f["S"] or i.f["A"] or i.f["uimm"]) for i in body):   # only nop
            return False
        loads = [i for i in body if i.op in ("lwz", "lhz", "lha", "lbz")]
        return bool(loads) and all(i.f["A"] in (2, 13) for i in loads)

    def jump(self, t, src):
        if self.entry <= t < self.unit.end:
            # a backward branch closes a loop: the safe point for interrupts
            if t > src:
                return f"goto L_{t:08X};"
            poll = "PPC_IDLE(c);" if self.idle_loop(t, src) else "PPC_POLL(c);"
            return f"{poll} goto L_{t:08X};"
        if t in self.entries:
            return f"{{ {E.fname(t)}(c); return; }}"
        return f'{{ ppc_unimplemented(c, 0x{t:08X}u, "branch to unknown target"); return; }}'

    def call(self, t):
        if t in self.entries:
            return f"{E.fname(t)}(c);"
        return f'ppc_unimplemented(c, 0x{t:08X}u, "call to unknown target");'


HOOKS_HEAD = """// Replaceable functions. Each has a slot the runtime can fill (ppc_hook);
// an empty slot runs the recompiled original, which stays callable. Also the
// data symbols the runtime looks up by name (ppc_symbol).
#include "funcs.h"

"""


def hook_names(img, files, log=print):
    """({addr: name} of hookable functions, [(name, addr)] of data symbols).

    A hook file lists one function name per line; `@name` asks for a data
    symbol's address instead; `#` starts a comment. Names the executable
    lacks are reported and skipped."""
    by_name = {}
    for s in img.symbols:
        if s.name and s.name not in by_name:
            by_name[s.name] = s
    hooks, syms = {}, []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                name = line.split("#")[0].strip()
                if not name:
                    continue
                data = name.startswith("@")
                name = name.lstrip("@")
                s = by_name.get(name)
                if s is None:
                    log(f"  hook: no symbol {name}")
                elif data:
                    syms.append((name, s.addr))
                else:
                    hooks[s.addr] = name
    return hooks, syms


def generate(img, out, per_file=8000, comments=True, hook_files=(), modules=(), log=print):
    t0 = time.time()
    prog = Program(img, log=log)
    entries = set(prog.entries)
    hooks, syms = hook_names(img, hook_files, log)
    missing = [a for a in hooks if a not in entries]
    for a in missing:
        log(f"  hook: {hooks.pop(a)} at {a:08X} is not a function entry")
    os.makedirs(out, exist_ok=True)
    unsupported, files, n_ins = {}, [], 0
    buf, count, idx = [], 0, 0

    def flush():
        nonlocal buf, count, idx
        if not buf:
            return
        name = f"recomp_{idx:03d}.cpp"
        with Out(os.path.join(out, name)) as f:
            f.write('#include "funcs.h"\n\n')
            f.write("\n".join(buf))
        files.append(name)
        buf, count, idx = [], 0, idx + 1

    for e in prog.entries:
        unit, body = prog.body(e)
        fn = FnCtx(prog, e, entries, next((m for m in modules if m.contains(e)), None))
        labels = prog.labels(e)
        title = unit.name if e == unit.start else f"{unit.name}+0x{e - unit.start:X}"
        cname = f"orig_{e:08X}" if e in hooks else E.fname(e)
        lines = [f"// {cw.demangle(title)}", f"void {cname}(PPCContext& c) {{"]
        for a, ins in body:
            if a in labels:
                lines.append(f"L_{a:08X}:")
            try:
                code = E.emit(ins, a, fn)
            except E.Unsupported as ex:
                unsupported[str(ex)] = unsupported.get(str(ex), 0) + 1
                code = [f'ppc_unimplemented(c, 0x{a:08X}u, "{ex}");']
            note = f"  // {a:08X} {_text(ins, a)}" if comments else ""
            lines.append("    " + code[0] + note)
            lines.extend("    " + x for x in code[1:])
        last = body[-1][1] if body else None
        if last is None or not E.terminates(last):
            lines.append("    " + fn.jump(unit.end, unit.end) if unit.end in entries else
                         f'    ppc_unimplemented(c, 0x{unit.end:08X}u, "fell off the end"); return;')
        lines.append("}\n")
        buf.append("\n".join(lines))
        count += len(body)
        n_ins += len(body)
        if count >= per_file:
            flush()
    flush()

    with Out(os.path.join(out, "funcs.h")) as f:
        f.write("#pragma once\n#include \"ppc.h\"\n\n")
        f.write("extern uint32_t g_rso_base[];   // where each RSO module is loaded (rso.cpp)\n\n")
        for e in prog.entries:
            f.write(f"void {E.fname(e)}(PPCContext& c);\n")
    with Out(os.path.join(out, "table.cpp")) as f:
        f.write('#include "funcs.h"\n#include <cstddef>\n\n')
        f.write("extern const PPCFuncEntry g_ppc_funcs[] = {\n")
        for e in prog.entries:
            f.write(f"    {{0x{e:08X}u, {E.fname(e)}}},\n")
        f.write("};\n")
        f.write(f"extern const size_t g_ppc_nfuncs = {len(prog.entries)};\n")
    with Out(os.path.join(out, "hooks.cpp")) as f:
        f.write(HOOKS_HEAD)
        for a, name in sorted(hooks.items()):
            f.write(f"// {name}\n"
                    f"void orig_{a:08X}(PPCContext& c);\n"
                    f"static PPCFunc hook_{a:08X};\n"
                    f"void {E.fname(a)}(PPCContext& c) {{ if (hook_{a:08X}) hook_{a:08X}(c); "
                    f"else orig_{a:08X}(c); }}\n\n")
        f.write("extern const PPCHook g_ppc_hooks[] = {\n")
        for a, name in sorted(hooks.items()):
            f.write(f'    {{"{name}", 0x{a:08X}u, orig_{a:08X}, &hook_{a:08X}}},\n')
        f.write("    {nullptr, 0, nullptr, nullptr}\n};\n")
        f.write("extern const PPCSymbol g_ppc_symbols[] = {\n")
        for name, a in syms:
            f.write(f'    {{"{name}", 0x{a:08X}u}},\n')
        f.write("    {nullptr, 0}\n};\n")
    with Out(os.path.join(out, "symbols.tsv")) as f:
        # the units as wiikit.dol --symbols writes them: names for the
        # runtime's logs (wiiboot --symbols), discovery's and the modules' too
        f.write("addr\tsize\tbind\ttype\tsection\tname\tdemangled\n")
        for u in prog.units:
            f.write(f"{u.start:08x}\t{u.end - u.start}\t1\t2\t.text\t{u.name}\t{cw.demangle(u.name)}\n")
    with Out(os.path.join(out, "rso.cpp")) as f:
        f.write('#include "funcs.h"\n#include <cstddef>\n\n'
                "// The RSO modules recompiled with the executable, at their virtual\n"
                "// addresses (wiikit/rso.py); the runtime fills g_rso_base as the game\n"
                "// links them (runtime/rso.cpp).\n")
        f.write("extern const RsoModule g_rso_modules[] = {\n")
        for m in modules:
            name = m.module.name.replace("\\", "\\\\")
            f.write(f'    {{"{name}", 0x{m.vbase:08X}u, 0x{m.size:X}u}},\n')
        f.write("    {nullptr, 0, 0}\n};\n")
        f.write(f"extern const size_t g_rso_nmodules = {len(modules)};\n")
        f.write(f"uint32_t g_rso_base[{max(1, len(modules))}];\n")
    with Out(os.path.join(out, "CMakeLists.txt")) as f:
        rt = RUNTIME.replace("\\", "/")
        srcs = "\n    ".join(files + ["table.cpp", "hooks.cpp", "rso.cpp"])
        f.write(f"""cmake_minimum_required(VERSION 3.20)
project(wiikit_recomp CXX)
set(CMAKE_CXX_STANDARD 20)
set(WIIKIT_RUNTIME "{rt}")
add_library(recomp STATIC
    {srcs})
target_include_directories(recomp PUBLIC ${{WIIKIT_RUNTIME}})
target_compile_options(recomp PRIVATE -Wno-unused-label -Wno-unused-variable)
# the runtime: wiikit_core (memory, dispatch, hooks), and either wiikit_stub
# (no hardware: tests) or wiikit_hw (the Wii replaced: boots the game)
include(${{WIIKIT_RUNTIME}}/runtime.cmake)
add_executable(linkcheck ${{WIIKIT_RUNTIME}}/linkcheck.cpp)
target_link_libraries(linkcheck wiikit_core wiikit_stub recomp)
# a project can add its own targets (tests, the game) with -DWIIKIT_EXTRA=file.cmake
if(DEFINED WIIKIT_EXTRA)
  include(${{WIIKIT_EXTRA}})
endif()
""")
    extra = [e for e in prog.entries if prog.unit_at(e).start != e]
    report = [
        f"units            {len(prog.units)} ({sum(1 for u in prog.units if u.name.startswith('unk_'))} gaps)",
        f"entries          {len(prog.entries)} ({len(extra)} inside units)",
        f"instructions     {n_ins}",
        f"switch tables    {len(prog.tables)} resolved, {prog.unresolved_tables} unresolved",
        f"bad targets      {len(prog.bad_targets)}",
        f"hooks            {len(hooks)} functions, {len(syms)} data symbols",
        f"unsupported      {unsupported or 'none'}",
        f"files            {len(files)}",
        f"time             {time.time() - t0:.1f} s",
    ]
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
        for a, t in prog.bad_targets:
            f.write(f"bad target {a:08X} -> {t:08X}\n")
    log("\n".join(report))


def _text(ins, a):
    from .. import ppc
    return ppc.text(ins, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-file", type=int, default=8000)
    ap.add_argument("--no-comments", action="store_true")
    ap.add_argument("--hooks", action="append", default=[],
                    help="an extra hook list; the runtime's own (runtime/hooks.txt) is always read")
    ap.add_argument("--sel", metavar="SEL",
                    help="the static module's export list, for --rso: the executable's names "
                         "the modules import")
    ap.add_argument("--rso", action="append", default=[], metavar="RSO",
                    help="an RSO module to recompile with the executable (repeatable; the "
                         "order numbers them)")
    ap.add_argument("--symbols", metavar="TSV",
                    help="names for a stripped executable (addr, ..., name); its functions "
                         "are found by discovery (recomp/discover.py)")
    a = ap.parse_args()
    img = Image(a.image)
    modules, module_names = [], {}
    if a.rso:
        from ..rso import attach
        modules, module_names = attach(img, a.rso, a.sel)
    if not img.functions() or modules:
        from .discover import load_names, symbolise
        names = load_names(a.symbols) if a.symbols else {}
        for v, n in module_names.items():
            names.setdefault(v, n)
        symbolise(img, names)
    generate(img, a.out, a.per_file, not a.no_comments,
             [os.path.join(RUNTIME, "hooks.txt")] + a.hooks, modules)


if __name__ == "__main__":
    main()
