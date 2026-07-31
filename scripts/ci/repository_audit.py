#!/usr/bin/env python3
"""Static GitHub-release audit for the adaptive-ensemble repository.

The audit intentionally uses only the Python standard library so it can run before Julia
packages are instantiated. It validates repository hygiene, active scope, preserved historical
Python/notebook hashes, TOML/JSON syntax, common credential patterns, and GitHub file-size limits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import py_compile
import re
import sys
import tempfile
import tomllib
from urllib.parse import unquote
from pathlib import Path

TEXT_SUFFIXES = {
    ".jl", ".py", ".ps1", ".psm1", ".psd1", ".sh", ".md", ".toml", ".yml",
    ".yaml", ".txt", ".cff", ".gitattributes", ".gitignore", ".csv",
}
MAX_GIT_BYTES = 100 * 1024 * 1024
WARN_GIT_BYTES = 50 * 1024 * 1024

REQUIRED_PATHS = [
    ".gitignore",
    ".gitattributes",
    ".github/workflows/ci.yml",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "DATA_AND_LICENSES.md",
    "docs/ARCHITECTURE.md",
    "docs/GITHUB_RELEASE.md",
    "docs/HISTORICAL_PIPELINE.md",
    "docs/SCIENTIFIC_SCOPE.md",
    "scripts/ci/repository_audit.sh",
    "scripts/ci/repository_audit.ps1",
    "Project.toml",
    "Manifest.toml",
    "src/AdaptiveEnsemblePublication.jl",
    "config/publication_full.toml",
    "config/publication_full_no_gurobi.toml",
    "test/runtests.jl",
    "reports/ORIGINAL_PYTHON_INTEGRITY.csv",
]

STALE_PATHS = [
    "STRESS_STRING_FIX_RECOVERY.md",
    "scripts/publication/migrate_after_stress_string_fix.jl",
    "scripts/publication/migrate_after_stress_string_fix.ps1",
    "reports/REMOVED_STALE_FILES_STRESS_STRING_FIX.csv",
    "reports/STATIC_VALIDATION_STRESS_STRING_FIX_2026-07-30.md",
    "reports/STRESS_STRING_CONVERSION_AUDIT_2026-07-30.md",
    "reports/SYNTHETIC_SAFI_SCOPE_AUDIT_2026-07-30.md",
    "reports/UPDATED_FILES_STRESS_STRING_FIX.csv",
]

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Gurobi WLS secret assignment": re.compile(
        r"(?i)\b(?:WLSSecret|WLSAccessID)\b\s*[=:]\s*[\"']?[A-Za-z0-9_-]{12,}"
    ),
}

ACTIVE_TEXT_ROOTS = [
    ".github",
    "config",
    "docs",
    "scripts/ci",
    "scripts/publication",
    "src/publication",
    "test",
]
ACTIVE_ROOT_FILES = [
    "README.md", "README_JULIA_PUBLICATION.md", "JULIA_PUBLICATION_RUNBOOK.md",
    "CODEX_DETACHED_RUNBOOK.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md",
    "DATA_AND_LICENSES.md", "CITATION.cff", "Project.toml",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def active_text_files(root: Path):
    seen: set[Path] = set()
    for relative in ACTIVE_ROOT_FILES:
        path = root / relative
        if path.is_file():
            seen.add(path)
            yield path
    for relative in ACTIVE_TEXT_ROOTS:
        base = root / relative
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path not in seen:
                seen.add(path)
                yield path


def check_active_scope(config_path: Path, errors: list[str]) -> None:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    real_datasets = config.get("real_world", {}).get("datasets", [])
    sensitivity_datasets = config.get("sensitivity", {}).get("datasets", [])
    if real_datasets != ["safi"]:
        errors.append(f"{config_path.name}: [real_world].datasets must be exactly ['safi']")
    if sensitivity_datasets != ["safi"]:
        errors.append(f"{config_path.name}: [sensitivity].datasets must be exactly ['safi']")
    scope = config.get("scope", {})
    if scope.get("label") != "synthetic_safi":
        errors.append(f"{config_path.name}: scope.label must be 'synthetic_safi'")
    if scope.get("paper_application_scope_complete") is not False:
        errors.append(
            f"{config_path.name}: paper_application_scope_complete must remain false"
        )
    fallback = config.get("adaptive", {}).get("irls_fallback_solver")
    if fallback not in (None, "none"):
        errors.append(
            f"{config_path.name}: large real-data IRLS fallback must be disabled, found {fallback!r}"
        )




def _balanced_delimiters(text: str, *, language: str) -> str | None:
    """Return a compact delimiter/string error, or None.

    This is intentionally a lexical guard rather than a parser. Julia execution in CI remains the
    authoritative syntax check; this catches truncated files, unclosed comments/strings, and broken
    PowerShell/Julia delimiter structure before dependency setup.
    """
    pairs = {')': '(', ']': '[', '}': '{'}
    stack: list[tuple[str, int]] = []
    i = 0
    n = len(text)
    block_comment_depth = 0
    quote: str | None = None
    triple = False
    here_end: str | None = None
    line = 1

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ''
        if ch == '\n':
            line += 1

        if here_end is not None:
            line_start = i == 0 or text[i - 1] == '\n'
            if line_start and text.startswith(here_end, i):
                i += len(here_end)
                here_end = None
                continue
            i += 1
            continue

        if block_comment_depth:
            if language == 'julia' and ch == '#' and nxt == '=':
                block_comment_depth += 1
                i += 2
                continue
            if language == 'julia' and ch == '=' and nxt == '#':
                block_comment_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if quote is not None:
            if ch == '\\':
                i += 2
                continue
            if triple and text.startswith(quote * 3, i):
                i += 3
                quote = None
                triple = False
                continue
            if not triple and ch == quote:
                i += 1
                quote = None
                continue
            i += 1
            continue

        if language == 'powershell' and text.startswith("@'", i):
            here_end = "'@"
            i += 2
            continue
        if language == 'powershell' and text.startswith('@"', i):
            here_end = '"@'
            i += 2
            continue
        if language == 'julia' and ch == '#' and nxt == '=':
            block_comment_depth = 1
            i += 2
            continue
        if ch == '#':
            newline = text.find('\n', i)
            i = n if newline < 0 else newline
            continue

        if ch == '"':
            triple = text.startswith('"""', i)
            quote = '"'
            i += 3 if triple else 1
            continue
        if language == 'julia' and ch == '`':
            quote = '`'
            i += 1
            continue
        if ch == "'":
            # A Julia transpose such as x' has no closing quote. Treat this as a character/string
            # literal only when a matching quote occurs before the line ends.
            end = i + 1
            escaped = False
            found = False
            while end < n and text[end] != '\n':
                if not escaped and text[end] == "'":
                    found = True
                    break
                escaped = (not escaped and text[end] == '\\')
                if text[end] != '\\':
                    escaped = False
                end += 1
            if found:
                i = end + 1
                continue

        if ch in '([{':
            stack.append((ch, line))
        elif ch in ')]}':
            if not stack or stack[-1][0] != pairs[ch]:
                return f"unexpected {ch!r} on line {line}"
            stack.pop()
        i += 1

    if block_comment_depth:
        return 'unterminated Julia block comment'
    if quote is not None:
        return f"unterminated {language} quoted literal"
    if here_end is not None:
        return 'unterminated PowerShell here-string'
    if stack:
        opener, opener_line = stack[-1]
        return f"unclosed {opener!r} opened on line {opener_line}"
    return None




def _check_markdown_links(root: Path, errors: list[str]) -> None:
    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in iter_files(root):
        if path.suffix.lower() != ".md":
            continue
        text = read_text(path)
        if text is None:
            continue
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip()
            if " \"" in target:
                target = target.split(" \"", 1)[0]
            target = target.strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(
                    f"Markdown link escapes repository in {path.relative_to(root).as_posix()}: {raw_target}"
                )
                continue
            if not resolved.exists():
                errors.append(
                    f"broken local Markdown link in {path.relative_to(root).as_posix()}: {raw_target}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        print(f"ERROR: repository root does not exist: {root}", file=sys.stderr)
        return 2

    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")
    for relative in STALE_PATHS:
        if (root / relative).exists():
            errors.append(f"stale one-off recovery artifact is still present: {relative}")

    gitignore = (root / ".gitignore").read_text(encoding="utf-8") if (root / ".gitignore").is_file() else ""
    if any(line.strip() == "*.csv" for line in gitignore.splitlines()):
        errors.append(".gitignore must not ignore every CSV; data and preserved results must be trackable")

    # Parse every TOML and notebook JSON file.
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size >= MAX_GIT_BYTES:
            errors.append(f"file exceeds GitHub's 100 MiB hard limit: {relative} ({size} bytes)")
        elif size >= WARN_GIT_BYTES:
            warnings.append(f"large Git object (>50 MiB): {relative} ({size} bytes)")
        if b"\x00" in path.read_bytes()[:1024 * 1024] and path.suffix.lower() in TEXT_SUFFIXES:
            errors.append(f"NUL byte in text file: {relative}")
        if path.suffix.lower() == ".toml":
            try:
                tomllib.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid TOML {relative}: {exc}")
        elif path.suffix.lower() == ".ipynb":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid notebook JSON {relative}: {exc}")

    # Historical Python files must remain byte-identical.
    integrity_path = root / "reports/ORIGINAL_PYTHON_INTEGRITY.csv"
    if integrity_path.is_file():
        with integrity_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 28:
            errors.append(f"Python/notebook integrity manifest has {len(rows)} rows; expected 28")
        for row in rows:
            relative = row["relative_path"]
            path = root / relative
            if not path.is_file():
                errors.append(f"preserved Python/notebook file missing: {relative}")
                continue
            if path.stat().st_size != int(row["bytes"]):
                errors.append(f"preserved file size changed: {relative}")
            if sha256(path) != row["sha256"]:
                errors.append(f"preserved file hash changed: {relative}")

    # Python syntax validation without writing caches into the repository.
    with tempfile.TemporaryDirectory(prefix="adaptive-ensemble-pyc-") as tmp:
        for path in sorted(root.rglob("*.py")):
            if ".git" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            try:
                target = Path(tmp) / (relative.replace("/", "__") + "c")
                py_compile.compile(str(path), cfile=str(target), doraise=True)
            except py_compile.PyCompileError as exc:
                errors.append(f"Python syntax error in {relative}: {exc.msg}")

    # Active code/docs must not contain merge markers, credentials, or user-specific paths.
    windows_user_pattern = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+")
    unix_user_pattern = re.compile(r"/(?:home|Users)/[^/\s]+")
    for path in active_text_files(root):
        relative = path.relative_to(root).as_posix()
        text = read_text(path)
        if text is None:
            continue
        if re.search(r"(?m)^(?:<<<<<<< |=======\s*$|>>>>>>> )", text):
            errors.append(f"merge-conflict marker in {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        if windows_user_pattern.search(text) or unix_user_pattern.search(text):
            errors.append(f"machine-specific user path in active file: {relative}")
        language = "julia" if path.suffix.lower() == ".jl" else (
            "powershell" if path.suffix.lower() in {".ps1", ".psm1", ".psd1"} else None
        )
        if language is not None:
            delimiter_error = _balanced_delimiters(text, language=language)
            if delimiter_error:
                errors.append(f"{language} lexical structure error in {relative}: {delimiter_error}")

    for config_name in ("publication_full.toml", "publication_full_no_gurobi.toml"):
        path = root / "config" / config_name
        if path.is_file():
            check_active_scope(path, errors)

    # Environment sanity: this is intentionally an application-style project.
    project = tomllib.loads((root / "Project.toml").read_text(encoding="utf-8"))
    if not project.get("deps"):
        errors.append("Project.toml has no [deps] table")
    manifest = tomllib.loads((root / "Manifest.toml").read_text(encoding="utf-8"))
    if manifest.get("julia_version") != "1.12.6":
        errors.append("Manifest.toml must record julia_version = '1.12.6'")

    _check_markdown_links(root, errors)

    forbidden_generated = [
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_dir() and (
            path.name == "publication_artifacts" or
            path.name.startswith("publication_artifacts_") or
            path.name.startswith("robust_norm_energy_diagnostic")
        )
    ]
    if forbidden_generated:
        errors.append("generated output directories committed at repository root: " + ", ".join(forbidden_generated))

    print(f"Repository: {root}")
    print(f"Files checked: {sum(1 for _ in iter_files(root))}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print("GITHUB REPOSITORY AUDIT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
