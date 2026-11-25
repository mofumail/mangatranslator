"""Static checks that need no models or heavy imports (run in CI).

- every `from mangatl.x import name` / `import mangatl.x` resolves to a module and a top-level name
- every attribute chain read off a Settings value exists on the Settings model
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [*sorted((ROOT / "mangatl").rglob("*.py")), ROOT / "translate.py", ROOT / "webui.py",
           *sorted((ROOT / "eval").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]


def _module_file(dotted: str) -> Path | None:
    p = ROOT.joinpath(*dotted.split("."))
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    return None


def _top_level_names(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.If, ast.Try)):  # e.g. guarded imports
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names |= {(a.asname or a.name).split(".")[0] for a in sub.names}
    return names


def _package_of(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    return ".".join(parts[:-1]) if parts[-1] != "__init__" else ".".join(parts[:-1])


def test_mangatl_imports_resolve():
    problems = []
    for src in SOURCES:
        for node in ast.walk(ast.parse(src.read_text())):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    base = _package_of(src).split(".")
                    base = base[: len(base) - node.level + 1]
                    mod = ".".join([*base, mod] if mod else base)
                if not mod.startswith("mangatl"):
                    continue
                target = _module_file(mod)
                if target is None:
                    problems.append(f"{src.relative_to(ROOT)}:{node.lineno}: no module {mod}")
                    continue
                names = _top_level_names(target)
                for a in node.names:
                    if a.name != "*" and a.name not in names and _module_file(f"{mod}.{a.name}") is None:
                        problems.append(f"{src.relative_to(ROOT)}:{node.lineno}: {mod} has no {a.name}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("mangatl") and _module_file(a.name) is None:
                        problems.append(f"{src.relative_to(ROOT)}:{node.lineno}: no module {a.name}")
    assert not problems, "\n".join(problems)


def _chain(node: ast.AST) -> list[str] | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


def test_settings_attribute_chains_exist():
    """config.a.b.c (and simple aliases of it) must exist on Settings, in every function that
    takes a Settings argument (named config/settings or annotated `Settings`)."""
    from mangatl.settings import Settings

    root_obj = Settings()
    problems = []
    for src in sorted((ROOT / "mangatl").rglob("*.py")):
        tree = ast.parse(src.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            roots: dict[str, list[str]] = {}
            for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
                ann = ast.unparse(a.annotation) if a.annotation is not None else ""
                if ann in ("Settings", "Settings | None", '"Settings"') or (not ann and a.arg in ("config", "settings")):
                    roots[a.arg] = []
            if not roots:
                continue
            for node in ast.walk(fn):  # simple aliases: name = config.x.y
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Attribute)):
                    c = _chain(node.value)
                    if c and c[0] in roots:
                        roots.setdefault(node.targets[0].id, roots[c[0]] + c[1:])
            for node in ast.walk(fn):
                if not isinstance(node, ast.Attribute):
                    continue
                c = _chain(node)
                if not c or c[0] not in roots:
                    continue
                obj = root_obj
                for attr in roots[c[0]] + c[1:]:
                    if not hasattr(obj, attr):
                        problems.append(f"{src.relative_to(ROOT)}:{node.lineno}: {'.'.join(c)} ({attr!r} missing)")
                        break
                    value = getattr(obj, attr)
                    if not hasattr(type(obj), "model_fields") or attr not in type(obj).model_fields:
                        break  # property / method / runtime object: stop following
                    obj = value
    assert not problems, "\n".join(sorted(set(problems)))
