"""The layering in AGENTS.md: core logic depends on nothing outside it but the catalogue's numpy."""

import ast
from pathlib import Path

CORE = Path("backend/core")
FORBIDDEN_IN_CORE = ("fastapi", "httpx", "uvicorn", "fastembed", "backend.adapters", "backend.api")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_imports_no_web_framework_http_client_or_adapter():
    leaks = {
        f"{path.name}: {module}"
        for path in CORE.glob("*.py")
        for module in imported_modules(path)
        if module.startswith(FORBIDDEN_IN_CORE)
    }
    assert leaks == set()
