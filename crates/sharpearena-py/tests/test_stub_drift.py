"""Two-way drift guard between the native module and its hand-written stub.

`sharpearena_py.pyi` is maintained by hand against the pyo3 source, and a stub
that rots is worse than none: a type checker trusts it over the runtime, so a
missing name reports a false error and a fabricated one blesses code that
crashes. Both directions are therefore asserted against the real compiled
module: every public runtime name must appear in the stub, and every stub name
must exist at runtime with the same callable/class/constant kind.
"""

from __future__ import annotations

import ast
import importlib.resources as resources
import inspect
from typing import Any

import pytest

native = pytest.importorskip("sharpearena.sharpearena_py")

STUB = resources.files("sharpearena").joinpath("sharpearena_py.pyi")


def _stub_kinds() -> dict[str, str]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    kinds: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kinds[node.name] = "function"
        elif isinstance(node, ast.ClassDef):
            kinds[node.name] = "class"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            kinds[node.target.id] = "constant"
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    kinds[target.id] = "constant"
    return kinds


def _runtime_kinds() -> dict[str, str]:
    kinds: dict[str, str] = {}
    for name in dir(native):
        if name.startswith("__") and name.endswith("__"):
            continue
        obj = getattr(native, name)
        if isinstance(obj, type):
            kinds[name] = "class"
        elif callable(obj):
            kinds[name] = "function"
        else:
            kinds[name] = "constant"
    return kinds


def _stub_class_members(node: ast.ClassDef) -> dict[str, str]:
    members: dict[str, str] = {}
    for member in node.body:
        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if member.name.startswith("__") and member.name.endswith("__"):
            continue
        decorators = {
            decorator.id
            for decorator in member.decorator_list
            if isinstance(decorator, ast.Name)
        }
        members[member.name] = "property" if "property" in decorators else "function"
    return members


def _stub_classes() -> dict[str, ast.ClassDef]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    return {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }


def _runtime_class_members(cls: type[Any]) -> dict[str, str]:
    members: dict[str, str] = {}
    for name, raw in vars(cls).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if inspect.isgetsetdescriptor(raw) or isinstance(raw, property):
            members[name] = "property"
        elif callable(getattr(cls, name)):
            members[name] = "function"
        else:
            members[name] = "constant"
    return members


_MISSING = object()


def _literal_default(node: ast.expr | None) -> object:
    if node is None:
        return _MISSING
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def _ast_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[tuple[str, str, object], ...]:
    positional = [*node.args.posonlyargs, *node.args.args]
    positional_defaults = [_MISSING] * (len(positional) - len(node.args.defaults)) + [
        _literal_default(default) for default in node.args.defaults
    ]
    parameters: list[tuple[str, str, object]] = []
    for argument, default in zip(positional, positional_defaults, strict=True):
        kind = "POSITIONAL_ONLY" if argument in node.args.posonlyargs else "POSITIONAL_OR_KEYWORD"
        parameters.append((argument.arg, kind, default))
    if node.args.vararg is not None:
        parameters.append((node.args.vararg.arg, "VAR_POSITIONAL", _MISSING))
    for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        parameters.append((argument.arg, "KEYWORD_ONLY", _literal_default(default)))
    if node.args.kwarg is not None:
        parameters.append((node.args.kwarg.arg, "VAR_KEYWORD", _MISSING))
    return tuple(parameters)


def _runtime_signature(obj: object) -> tuple[tuple[str, str, object], ...]:
    return tuple(
        (
            parameter.name,
            parameter.kind.name,
            _MISSING if parameter.default is inspect.Parameter.empty else parameter.default,
        )
        for parameter in inspect.signature(obj).parameters.values()
    )


def _without_receiver(
    parameters: tuple[tuple[str, str, object], ...],
) -> tuple[tuple[str, str, object], ...]:
    if parameters and parameters[0][0] in {"self", "cls"}:
        return parameters[1:]
    return parameters


def test_every_runtime_name_appears_in_the_stub() -> None:
    missing = sorted(set(_runtime_kinds()) - set(_stub_kinds()))
    assert not missing, (
        f"the native module exposes {missing} but sharpearena_py.pyi does not "
        "declare them, so a type checker reports them as nonexistent"
    )


def test_every_stub_name_exists_at_runtime() -> None:
    fabricated = sorted(set(_stub_kinds()) - set(_runtime_kinds()))
    assert not fabricated, (
        f"sharpearena_py.pyi declares {fabricated} but the native module does "
        "not expose them, so a type checker blesses calls that crash"
    )


def test_shared_names_agree_on_callable_class_constant_kind() -> None:
    stub = _stub_kinds()
    runtime = _runtime_kinds()
    disagreements = sorted(
        f"{name}: stub says {stub[name]}, runtime is {runtime[name]}"
        for name in set(stub) & set(runtime)
        if stub[name] != runtime[name]
    )
    assert not disagreements, disagreements


def test_native_classes_and_the_stub_have_the_same_public_members() -> None:
    stub_classes = _stub_classes()
    disagreements: list[str] = []
    for class_name, stub_node in stub_classes.items():
        runtime_class = getattr(native, class_name)
        stub_members = _stub_class_members(stub_node)
        runtime_members = _runtime_class_members(runtime_class)
        missing = sorted(set(runtime_members) - set(stub_members))
        fabricated = sorted(set(stub_members) - set(runtime_members))
        wrong_kind = sorted(
            name
            for name in set(stub_members) & set(runtime_members)
            if stub_members[name] != runtime_members[name]
        )
        if missing:
            disagreements.append(f"{class_name}: runtime-only members {missing}")
        if fabricated:
            disagreements.append(f"{class_name}: stub-only members {fabricated}")
        if wrong_kind:
            disagreements.append(
                f"{class_name}: property/function kind differs for {wrong_kind}"
            )
    assert not disagreements, disagreements


def test_callable_parameter_names_kinds_and_defaults_match_runtime() -> None:
    """pyo3 exposes its Python signatures, so a renamed, reordered, or re-defaulted
    parameter must make the hand-written stub fail. Type annotations remain a review
    surface because the native module carries no runtime type metadata."""

    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    disagreements: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not hasattr(native, node.name):
                # The existence test owns this diagnostic; avoid obscuring it with
                # an AttributeError before the full drift report is printed.
                continue
            stub_signature = _without_receiver(_ast_signature(node))
            runtime_signature = _without_receiver(
                _runtime_signature(getattr(native, node.name))
            )
            if stub_signature != runtime_signature:
                disagreements.append(
                    f"{node.name}: stub {stub_signature!r}, runtime {runtime_signature!r}"
                )
            continue
        if not isinstance(node, ast.ClassDef) or not hasattr(native, node.name):
            continue
        runtime_class = getattr(native, node.name)
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {
                decorator.id
                for decorator in member.decorator_list
                if isinstance(decorator, ast.Name)
            }
            if "property" in decorators:
                continue
            runtime_callable = (
                runtime_class
                if member.name == "__init__"
                else getattr(runtime_class, member.name)
            )
            stub_signature = _without_receiver(_ast_signature(member))
            runtime_signature = _without_receiver(_runtime_signature(runtime_callable))
            if stub_signature != runtime_signature:
                disagreements.append(
                    f"{node.name}.{member.name}: stub {stub_signature!r}, "
                    f"runtime {runtime_signature!r}"
                )
    assert not disagreements, disagreements
