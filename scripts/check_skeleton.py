#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_python_files() -> None:
    files = [ROOT / "setup.py"]
    files.extend((ROOT / "python").glob("**/*.py"))
    files.extend((ROOT / "benchmarks").glob("**/*.py"))
    files.extend((ROOT / "tests").glob("**/*.py"))
    files.extend((ROOT / "scripts").glob("**/*.py"))

    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def extract_setup_sources() -> list[str]:
    setup_py = ROOT / "setup.py"
    tree = ast.parse(setup_py.read_text(encoding="utf-8"), filename=str(setup_py))
    sources: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "sources":
            continue
        if not isinstance(node.value, ast.List):
            fail("setup.py CUDAExtension sources must be a literal list")
        for elt in node.value.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                fail("setup.py CUDAExtension sources must use relative string literals")
            sources.append(elt.value)

    if not sources:
        fail("no CUDAExtension sources found in setup.py")

    return sources


def check_setup_sources() -> None:
    for source in extract_setup_sources():
        path = Path(source)
        if path.is_absolute():
            fail(f"source path must be relative for editable install: {source}")
        if not (ROOT / path).is_file():
            fail(f"source listed in setup.py does not exist: {source}")


def check_project_includes() -> None:
    include_re = re.compile(r'^\s*#include\s+"([^"]+)"')
    for path in CSRC.glob("**/*"):
        if path.suffix not in {".h", ".cuh", ".cpp", ".cu"}:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = include_re.match(line)
            if not match:
                continue
            include = match.group(1)
            candidates = [
                path.parent / include,
                CSRC / include,
            ]
            if not any(candidate.is_file() for candidate in candidates):
                rel = path.relative_to(ROOT)
                fail(f"missing project include {include!r} referenced by {rel}:{lineno}")


def collect_declared_functions() -> set[str]:
    pattern = re.compile(r"torch::Tensor\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    declared: set[str] = set()
    for path in CSRC.glob("**/*.h"):
        declared.update(pattern.findall(path.read_text(encoding="utf-8")))
    return declared


def collect_defined_functions() -> set[str]:
    pattern = re.compile(r"torch::Tensor\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    defined: set[str] = set()
    for path in CSRC.glob("**/*.cu"):
        defined.update(pattern.findall(path.read_text(encoding="utf-8")))
    return defined


def collect_bound_functions() -> set[str]:
    bindings = ROOT / "csrc" / "bindings.cpp"
    pattern = re.compile(r"m\.def\([^,]+,\s*&([A-Za-z_][A-Za-z0-9_]*)")
    return set(pattern.findall(bindings.read_text(encoding="utf-8")))


def check_bindings() -> None:
    declared = collect_declared_functions()
    defined = collect_defined_functions()
    bound = collect_bound_functions()

    for fn in sorted(bound - declared):
        fail(f"function bound in bindings.cpp but not declared in a header: {fn}")
    for fn in sorted(bound - defined):
        fail(f"function bound in bindings.cpp but not defined in a .cu file: {fn}")


def main() -> None:
    parse_python_files()
    check_setup_sources()
    check_project_includes()
    check_bindings()
    print("skeleton check passed")


if __name__ == "__main__":
    main()

