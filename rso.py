"""RSO: the Wii SDK's relocatable modules, and the static module's export list (.sel).

A game built with the SDK's RSO library ships part of its code as `.rso`
files, loaded into the heap at run time and linked against the executable
(the "static module", whose exported names are in a `.sel` file) and
against each other, by name. Unlike a GameCube REL, an RSO keeps its
symbol names: every export and import is a string.

Layout (all big-endian, offsets from the file's start):

    0x00  next, prev                    the loaded list (0 on disc)
    0x08  section count, section table  (offset, size) pairs; offset 0 = none,
                                        low bit 1 = executable
    0x10  name offset, name size        the module's name (not NUL-terminated)
    0x18  version
    0x1C  bss size
    0x20  prolog, epilog, unresolved, bss section indices (u8 each)
    0x24  prolog, epilog, unresolved    offsets in their sections
    0x30  internal relocations          offset, length (bytes)
    0x38  external relocations          offset, length
    0x40  exports                       table offset, length, names offset
    0x4C  imports                       table offset, length, names offset

A relocation is (offset, info, addend): `offset` from the module's start
(the patched word's file offset), `info` = symbol << 8 | type (ELF PPC
types), the symbol a section index for internal ones, an import index for
external ones. An export is (name, offset in its section, section, hash);
an import (name, 0, the offset of its first external relocation).

    python -m wiikit.rso MODULE.rso [...]      # header, sections, counts
    python -m wiikit.rso MODULE.rso --exports | --imports | --relocs
"""
import collections
import struct
import sys

R_PPC = {0: "NONE", 1: "ADDR32", 2: "ADDR24", 3: "ADDR16", 4: "ADDR16_LO", 5: "ADDR16_HI",
         6: "ADDR16_HA", 7: "ADDR14", 8: "ADDR14_BRTAKEN", 9: "ADDR14_BRNTAKEN", 10: "REL24",
         11: "REL14", 12: "REL14_BRTAKEN", 13: "REL14_BRNTAKEN", 26: "REL32",
         201: "RVL_NONE", 202: "RVL_SECT", 203: "RVL_STOP"}


class Section:
    __slots__ = ("index", "offset", "size", "exec")

    def __init__(self, index, offset, size, exe):
        self.index, self.offset, self.size, self.exec = index, offset, size, exe

    def __repr__(self):
        return f"<sec {self.index} @{self.offset:#x} {self.size:#x}{' X' if self.exec else ''}>"


class Reloc:
    __slots__ = ("offset", "type", "symbol", "addend")

    def __init__(self, offset, info, addend):
        self.offset, self.type, self.symbol, self.addend = offset, info & 0xFF, info >> 8, addend

    @property
    def type_name(self):
        return R_PPC.get(self.type, str(self.type))


class Export:
    __slots__ = ("name", "offset", "section", "hash")

    def __init__(self, name, offset, section, h):
        self.name, self.offset, self.section, self.hash = name, offset, section, h


class Import:
    __slots__ = ("name", "offset", "reloc")

    def __init__(self, name, offset, reloc):
        self.name, self.offset, self.reloc = name, offset, reloc


def elf_hash(name):
    """The hash an export carries (ELF's)."""
    h = 0
    for ch in name.encode():
        h = ((h << 4) + ch) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h


class Module:
    def __init__(self, data, path=""):
        self.data, self.path = data, path
        u32 = lambda o: struct.unpack_from(">I", data, o)[0]
        self.u32 = u32
        nsec, sectab = u32(0x08), u32(0x0C)
        name_off, name_len = u32(0x10), u32(0x14)
        self.version = u32(0x18)
        self.name = data[name_off:name_off + name_len].decode("ascii", "replace") if name_off else ""
        self.bss_size = u32(0x1C)
        self.prolog_section, self.epilog_section, self.unresolved_section, self.bss_section = data[0x20:0x24]
        self.prolog, self.epilog, self.unresolved = u32(0x24), u32(0x28), u32(0x2C)
        self.sections = []
        for i in range(nsec):
            off, size = u32(sectab + 8 * i), u32(sectab + 8 * i + 4)
            self.sections.append(Section(i, off & ~1, size, bool(off & 1)))
        self.internal = self._relocs(u32(0x30), u32(0x34))
        self.external = self._relocs(u32(0x38), u32(0x3C))
        exp_off, exp_len, exp_names = u32(0x40), u32(0x44), u32(0x48)
        imp_off, imp_len, imp_names = u32(0x4C), u32(0x50), u32(0x54)
        self.exports = [Export(self._str(exp_names + u32(exp_off + k)), u32(exp_off + k + 4),
                               u32(exp_off + k + 8), u32(exp_off + k + 12))
                        for k in range(0, exp_len, 16)]
        self.imports = [Import(self._str(imp_names + u32(imp_off + k)), u32(imp_off + k + 4),
                               u32(imp_off + k + 8))
                        for k in range(0, imp_len, 12)]

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(f.read(), path)

    def _relocs(self, off, length):
        return [Reloc(*struct.unpack_from(">III", self.data, off + k)) for k in range(0, length, 12)]

    def _str(self, off):
        end = self.data.index(b"\0", off)
        return self.data[off:end].decode("ascii", "replace")

    @property
    def is_static(self):
        """A .sel: the executable's exports, sections without contents."""
        return all(s.offset == 0 for s in self.sections)

    def section_of(self, off):
        """The section holding file offset `off`, or None."""
        for s in self.sections:
            if s.offset and s.offset <= off < s.offset + s.size:
                return s
        return None


# --- recompiling modules -------------------------------------------------------------
#
# A module's code is recompiled with the executable's, at a virtual address:
# each module is linked where no guest address can be (from VIRTUAL_BASE, just
# past MEM1's 24 MB, within a `bl`'s reach of the executable's code), and its
# sections become segments of the executable's image. Discovery, switch
# tables, direct calls into the executable all work there as for the
# executable's own code. At run time the module is wherever the game loaded
# it; the generated code does not depend on that:
#
#   * an instruction whose 16-bit immediate is relocated (lis/addi/lwz... on
#     a symbol) reads the immediate from the loaded module, which the SDK's
#     own linker has patched: `imm_sites`;
#   * a `bl` to the executable or inside the module is a direct call, as
#     linked here;
#   * an address the code computes (a switch's case, a virtual method) is a
#     run-time address: the runtime turns it into the virtual one, knowing
#     where each module was loaded (runtime/rso.cpp, from RSOLinkListFixed).
#
# Return addresses stay virtual: the code only returns through them.

VIRTUAL_BASE = 0x81800000
_REACH = 0x02000000                       # a `bl`'s: +-32 MB


class Linked:
    """One module as recompiled: where it is linked, and what the code needs."""

    def __init__(self, index, module, vbase):
        self.index, self.module, self.vbase = index, module, vbase
        self.size = len(module.data)
        bss = next((s for s in module.sections if s.size and not s.offset), None)
        self.bss_section = bss.index if bss else None
        self.bss_vaddr = (vbase + self.size + 31) & ~31
        self.end = self.bss_vaddr + (bss.size if bss else 0)
        self.text_sections = {module.prolog_section, module.epilog_section,
                              module.unresolved_section} - {0}
        self.imm_sites = set()            # vaddrs of instructions whose immediate is relocated

    def section_vaddr(self, i):
        s = self.module.sections[i]
        if s.offset:
            return self.vbase + s.offset
        return self.bss_vaddr if i == self.bss_section else None

    def contains(self, a):
        return self.vbase <= a < self.end


def static_exports(img, sel):
    """{name: address} of the static module's (.sel) exports that can be
    placed: its sections 1 and 2 are .init and .text, the executable's first
    two text segments (CodeWarrior's order); absolute symbols are their value.
    Data sections have no address in the .sel: only the SDK's linker knows
    them at run time."""
    bases = {i + 1: s.vaddr for i, s in enumerate(img.text_segments()[:2])}
    out = {}
    for e in sel.exports:
        if e.section in bases:
            out[e.name] = bases[e.section] + e.offset
        elif e.section == 0xFFF1:
            out[e.name] = e.offset
    return out


def attach(img, paths, sel_path, log=print):
    """Link the modules at `paths` into `img` (a wiikit.dol.Image) at virtual
    addresses; returns ([Linked], {vaddr: name} of their exports)."""
    from .dol import Segment
    statics = static_exports(img, Module.load(sel_path)) if sel_path else {}
    linked, names = [], {}
    cursor = VIRTUAL_BASE
    for k, p in enumerate(paths):
        m = Module.load(p)
        L = Linked(k, m, cursor)
        cursor = (L.end + 0xFFF) & ~0xFFF
        data = bytearray(m.data)
        unresolved = set()
        for kind, rs in (("int", m.internal), ("ext", m.external)):
            for r in rs:
                if kind == "int":
                    S = L.section_vaddr(r.symbol)
                else:
                    S = statics.get(m.imports[r.symbol].name)
                    if S is None:
                        if r.type in (10, 11):    # a branch must land somewhere known
                            raise ValueError(f"{p}: a branch to {m.imports[r.symbol].name}, "
                                             "which the .sel does not place")
                        unresolved.add(m.imports[r.symbol].name)
                sec = m.section_of(r.offset)
                in_text = sec is not None and sec.index in L.text_sections
                if r.type in (4, 5, 6, 3) and in_text:      # a 16-bit immediate
                    L.imm_sites.add(L.vbase + r.offset - 2)
                if S is None:
                    continue                  # read at run time; nothing to fold here
                v = (S + r.addend) & 0xFFFFFFFF
                site = L.vbase + r.offset
                if r.type == 1:
                    struct.pack_into(">I", data, r.offset, v)
                elif r.type in (3, 4):
                    struct.pack_into(">H", data, r.offset, v & 0xFFFF)
                elif r.type == 5:
                    struct.pack_into(">H", data, r.offset, v >> 16)
                elif r.type == 6:
                    struct.pack_into(">H", data, r.offset, ((v + 0x8000) >> 16) & 0xFFFF)
                elif r.type == 10:
                    d = (v - site) & 0xFFFFFFFF
                    if not (d < _REACH or d >= 0x100000000 - _REACH):
                        raise ValueError(f"{p}: bl at {site:08X} cannot reach {v:08X}")
                    w = struct.unpack_from(">I", data, r.offset)[0]
                    struct.pack_into(">I", data, r.offset, (w & 0xFC000003) | (d & 0x03FFFFFC))
                elif r.type == 11:
                    d = (v - site) & 0xFFFFFFFF
                    w = struct.unpack_from(">I", data, r.offset)[0]
                    struct.pack_into(">I", data, r.offset, (w & 0xFFFF0003) | (d & 0xFFFC))
                else:
                    raise ValueError(f"{p}: relocation type {r.type_name} not handled")
        for s in m.sections:
            if s.offset and s.size:
                img.segments.append(Segment(L.vbase + s.offset, bytes(data[s.offset:s.offset + s.size]),
                                            s.size, s.index in L.text_sections,
                                            f"{module_name(m)}:{s.index}"))
        for e in m.exports:
            a = L.section_vaddr(e.section)
            if a is not None and e.section in L.text_sections:
                names.setdefault(a + e.offset, e.name)
        text = sum(m.sections[i].size for i in L.text_sections)
        log(f"  rso {k:2}: {module_name(m):16} at {L.vbase:08X}, {text:#x} bytes of code, "
            f"{len(L.imm_sites)} relocated immediates"
            + (f", {len(unresolved)} data imports (placed at run time)" if unresolved else ""))
        linked.append(L)
    img.segments.sort(key=lambda s: s.vaddr)
    img._starts = [s.vaddr for s in img.segments]
    lo = min(s.vaddr for s in img.text_segments())
    if linked and linked[-1].end - lo > _REACH:
        raise ValueError(f"the modules end at {linked[-1].end:08X}, beyond a bl's reach of {lo:08X}")
    return linked, names


def module_name(m):
    """The module's short name: its file name without the build's path and extension."""
    n = m.name.replace("\\", "/").rsplit("/", 1)[-1]
    return n.rsplit(".", 1)[0] if "." in n else n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m wiikit.rso", description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exports", action="store_true")
    ap.add_argument("--imports", action="store_true")
    ap.add_argument("--relocs", action="store_true")
    a = ap.parse_args(argv)
    for p in a.files:
        m = Module.load(p)
        kind = "static module (.sel)" if m.is_static else "module"
        print(f"{p}: {kind} '{m.name}' v{m.version}, {len(m.sections)} sections, bss {m.bss_size:#x}")
        for s in m.sections:
            if s.offset or s.size:
                print(f"  {s.index:2}  @{s.offset:08X}  {s.size:8X}  {'text' if s.exec else 'data'}")
        if not m.is_static:
            for what, sec, off in (("prolog", m.prolog_section, m.prolog),
                                   ("epilog", m.epilog_section, m.epilog),
                                   ("unresolved", m.unresolved_section, m.unresolved)):
                print(f"  {what}: section {sec} + {off:#x}")
        cnt = collections.Counter(r.type_name for r in m.internal)
        cnt_x = collections.Counter(r.type_name for r in m.external)
        print(f"  relocations: {len(m.internal)} internal {dict(cnt)}")
        print(f"               {len(m.external)} external {dict(cnt_x)}")
        print(f"  exports {len(m.exports)}, imports {len(m.imports)}")
        if a.exports:
            for e in m.exports:
                print(f"    E  {e.section:2} + {e.offset:08X}  {e.name}")
        if a.imports:
            for i in m.imports:
                print(f"    I  {i.name}  (first reloc @{i.reloc:#x})")
        if a.relocs:
            for kind, rs in (("int", m.internal), ("ext", m.external)):
                for r in rs:
                    print(f"    {kind} @{r.offset:08X} {r.type_name:12} sym {r.symbol:4} + {r.addend:08X}")


if __name__ == "__main__":
    main()
