"""Append one class to the end of a tool class list file (Models/<Model>.yaml), touching nothing else.

The files are hand-written YAML with comments and roboflow metadata, so they are never re-dumped:
the new name is inserted in front of the closing bracket of the `names:` list (and `nc:` is raised
when it matched the old count). The result is read back and must equal the old file plus exactly
one name at the end; otherwise nothing is written. Writing is a temp file + rename in the same
folder, so the tool never sees half a file. A copy of the old file is kept for rollback.
"""
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

NAME_RE = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,39}$")


class ClassListError(Exception):
    """The class list cannot be extended safely; nothing was written. The message is safe to show."""


def clean_name(value: Any) -> str:
    name = "_".join(str(value or "").strip().upper().split())
    if not NAME_RE.match(name):
        raise ClassListError("Ime klase: velika slova, brojevi, _ ili -, od 2 do 40 znakova (npr. MATCHA_CASA).")
    return name


def find_file(models_dir: Path, model_key: str) -> Path:
    for suffix in (".yaml", ".yml"):
        path = models_dir / f"{model_key}{suffix}"
        if path.is_file():
            return path
    raise ClassListError(f"Lista klasa alata za model {model_key} nije pronađena.")


def _read_names(data: Any) -> List[str]:
    names = (data or {}).get("names") if isinstance(data, dict) else None
    if isinstance(names, dict):
        return [v for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if isinstance(names, list):
        return list(names)
    raise ClassListError("Lista klasa alata nema ispravan spisak names.")


def _closing_bracket(text: str, start: int) -> Tuple[int, int]:
    """(index of the ']' that closes the flow list opened at text[start] == '[', index of the last item character)."""
    depth, i, last = 0, start, start
    quote: Optional[str] = None
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                if quote == "'" and text[i + 1:i + 2] == "'":
                    i += 1
                else:
                    quote = None
            elif c == "\\" and quote == '"':
                i += 1
            last = i
        elif c == "#" and (i == 0 or text[i - 1] in " \t\n"):
            nl = text.find("\n", i)
            i = len(text) if nl < 0 else nl
            continue
        elif c in "'\"":
            quote, last = c, i
        elif c == "[":
            depth += 1
            last = i
        elif c == "]":
            depth -= 1
            if depth == 0:
                return i, last
            last = i
        elif not c.isspace():
            last = i
        i += 1
    raise ClassListError("Lista klasa alata nije zatvorena (nedostaje ]).")


def appended_text(text: str, name: str) -> str:
    """The same file text with name added as the last class. Raises when the file is not in a form this understands."""
    match = re.search(r"^names:[ \t]*\[", text, re.MULTILINE)
    if match is None:
        raise ClassListError("Lista klasa alata nije u obliku names: [ ... ]. Dopiši klasu ručno, pa pokreni uvoz.")
    close, last = _closing_bracket(text, match.end() - 1)
    line_start = text.rfind("\n", 0, close) + 1
    own_line = text[line_start:close].strip() == ""
    item = f"'{name}'"
    if text[last] == "[":
        new = text[:last + 1] + item + text[last + 1:]
    elif own_line:
        prev_line = text[text.rfind("\n", 0, last) + 1:]
        indent = prev_line[:len(prev_line) - len(prev_line.lstrip())]
        head = text[:last + 1] + ("" if text[last] == "," else ",")
        trailing = "," if text[last] == "," else ""
        new = head + text[last + 1:line_start] + f"{indent}{item}{trailing}\n" + text[line_start:]
    elif text[last] == ",":
        new = text[:last + 1] + f" {item}," + text[last + 1:]
    else:
        new = text[:last + 1] + f", {item}" + text[last + 1:]
    return new


def append_class(path: Path, name: str, expected: List[str], backup_dir: Path) -> Path:
    """Append name to the class list file at path. expected = the names the file must hold now (Catalog order).

    Returns the backup copy of the old file. Raises ClassListError and writes nothing when the file
    disagrees with expected or the edit would change anything besides adding the one name (and nc).
    """
    old_text = path.read_text(encoding="utf-8")
    old_data = yaml.safe_load(old_text) or {}
    old_names = _read_names(old_data)
    if old_names != list(expected):
        raise ClassListError("Lista klasa alata i Katalog se ne slažu. Pokreni uvoz klasa, pa pokušaj ponovo.")
    if name in old_names:
        raise ClassListError(f"Klasa {name} već postoji u ovom modelu.")
    new_text = appended_text(old_text, name)
    nc = old_data.get("nc") if isinstance(old_data, dict) else None
    if isinstance(nc, int) and nc == len(old_names):
        new_text, count = re.subn(rf"^nc:([ \t]*){nc}([ \t]*(#.*)?)$", rf"nc:\g<1>{nc + 1}\g<2>", new_text, count=1, flags=re.MULTILINE)
        if count != 1:
            raise ClassListError("Broj klasa (nc) u listi alata ne može da se ažurira. Dopiši klasu ručno.")
    try:
        new_data = yaml.safe_load(new_text) or {}
    except yaml.YAMLError:
        raise ClassListError("Dopisivanje bi pokvarilo listu klasa alata. Ništa nije promenjeno.")
    expected_data: Dict[str, Any] = dict(old_data)
    if isinstance(nc, int) and nc == len(old_names):
        expected_data["nc"] = nc + 1
    expected_data.pop("names", None)
    got = dict(new_data) if isinstance(new_data, dict) else {}
    got_names = _read_names(got)
    got.pop("names", None)
    if got_names != old_names + [name] or got != expected_data:
        raise ClassListError("Dopisivanje bi promenilo još nešto u listi klasa alata. Ništa nije promenjeno.")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = backup_dir / f"{path.stem}.{stamp}{path.suffix}.bak"
    shutil.copy2(path, backup)
    tmp = path.with_name(f".{path.name}.katalog-tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_text)
        f.flush()
        os.fsync(f.fileno())
    shutil.copymode(path, tmp)
    os.replace(tmp, path)
    return backup


def restore(path: Path, backup: Path) -> None:
    """Put the old file back (used when the Catalog could not record the class after the file was written)."""
    tmp = path.with_name(f".{path.name}.katalog-tmp")
    shutil.copy2(backup, tmp)
    os.replace(tmp, path)
