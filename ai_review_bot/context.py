from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .diff_parser import ChangedFile


@dataclass(frozen=True)
class FunctionContext:
    file_path: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    source: str


def _qualified_functions(tree: ast.AST, source_lines: list[str], file_path: str) -> list[FunctionContext]:
    contexts: list[FunctionContext] = []
    class_stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            class_stack.append(node.name)
            for child in node.body:
                visit(child)
            class_stack.pop()
            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = getattr(node, "end_lineno", node.lineno)
            source = "\n".join(source_lines[node.lineno - 1 : end_line])
            qualified = ".".join([*class_stack, node.name]) if class_stack else node.name
            contexts.append(
                FunctionContext(
                    file_path=file_path,
                    name=node.name,
                    qualified_name=qualified,
                    start_line=node.lineno,
                    end_line=end_line,
                    source=source,
                )
            )
            for child in node.body:
                visit(child)
            return

        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return contexts


def get_python_functions(repo_root: Path, relative_path: str) -> list[FunctionContext]:
    file_path = repo_root / relative_path
    if file_path.suffix != ".py" or not file_path.exists():
        return []
    source = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _qualified_functions(tree, source.splitlines(), relative_path)


def locate_changed_functions(repo_root: Path, changed_file: ChangedFile) -> list[FunctionContext]:
    changed_lines = changed_file.changed_lines
    if not changed_lines:
        return []
    return [
        fn
        for fn in get_python_functions(repo_root, changed_file.path)
        if any(fn.start_line <= line <= fn.end_line for line in changed_lines)
    ]

