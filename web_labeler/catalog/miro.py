"""Old Miro boards: every PDF page becomes one whole image. Pages are never cut into articles.

Only reads the PDFs. Extraction runs in a temporary folder; the Catalog stores the result itself.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

_PAGE_KEY = re.compile(r"^[a-z0-9_]+-\d{2,3}$")


class MiroError(Exception):
    pass


@dataclass
class MiroPage:
    key: str
    model: str
    source_file: str
    page_no: int
    data: bytes
    sha256: str
    width: int
    height: int


@dataclass
class Suggestion:
    heading: str
    classes: List[str]
    confidence: str = "normal"
    note: str = ""


@dataclass
class MiroPlan:
    pages: List[MiroPage]
    suggestions: Dict[str, Suggestion] = field(default_factory=dict)
    report: Dict[str, object] = field(default_factory=dict)


def page_key(model: str, page_no: int) -> str:
    return f"{model.lower()}-{page_no:02d}"


def _run(args: Sequence[str]) -> str:
    try:
        done = subprocess.run(list(args), check=True, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise MiroError(f"{args[0]} is not installed (package poppler-utils)")
    except subprocess.CalledProcessError as e:
        raise MiroError(f"{args[0]} failed: {e.stderr.strip() or e}")
    except subprocess.TimeoutExpired:
        raise MiroError(f"{args[0]} took too long")
    return done.stdout


def _page_count(pdf: Path) -> int:
    match = re.search(r"^Pages:\s+(\d+)", _run(["pdfinfo", str(pdf)]), re.M)
    if not match:
        raise MiroError(f"{pdf.name}: cannot read the number of pages")
    return int(match.group(1))


def _images_per_page(pdf: Path) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for line in _run(["pdfimages", "-list", str(pdf)]).splitlines()[2:]:
        parts = line.split()
        if len(parts) > 2 and parts[0].isdigit() and parts[2] == "image":
            counts[int(parts[0])] = counts.get(int(parts[0]), 0) + 1
    return counts


def _jpeg_size(data: bytes) -> Tuple[int, int]:
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(data)) as im:
        im.verify()
    with Image.open(BytesIO(data)) as im:
        return im.size


def extract_pdf(pdf: Path, model: str) -> List[MiroPage]:
    """One image per page. A page that is a single JPEG is kept byte for byte; any other page is rendered whole."""
    if not pdf.is_file():
        raise MiroError(f"Miro file is missing: {pdf}")
    pages = _page_count(pdf)
    per_page = _images_per_page(pdf)
    result: List[MiroPage] = []
    with tempfile.TemporaryDirectory(prefix="catalog_miro_") as tmp:
        out = Path(tmp)
        _run(["pdfimages", "-j", "-p", str(pdf), str(out / "img")])
        for page_no in range(1, pages + 1):
            found = sorted(out.glob(f"img-{page_no:03d}-*.jpg"))
            if per_page.get(page_no) == 1 and len(found) == 1:
                data = found[0].read_bytes()
            else:
                base = out / f"page-{page_no:03d}"
                _run(["pdftoppm", "-r", "144", "-jpeg", "-singlefile", "-f", str(page_no), "-l", str(page_no),
                      str(pdf), str(base)])
                data = base.with_suffix(".jpg").read_bytes()
            try:
                width, height = _jpeg_size(data)
            except Exception as e:
                raise MiroError(f"{pdf.name} page {page_no}: not a readable image ({e})")
            result.append(MiroPage(page_key(model, page_no), model, pdf.name, page_no, data,
                                   hashlib.sha256(data).hexdigest(), width, height))
    return result


def read_suggestions(path: Path) -> Dict[str, Suggestion]:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise MiroError(f"{path.name}: cannot read suggestions ({e})")
    pages = doc.get("pages")
    if not isinstance(pages, dict):
        raise MiroError(f"{path.name}: expected a 'pages' mapping")
    result: Dict[str, Suggestion] = {}
    for key, item in pages.items():
        if not _PAGE_KEY.match(str(key)) or not isinstance(item, dict):
            raise MiroError(f"{path.name}: bad entry {key!r}")
        classes = item.get("classes") or []
        if not isinstance(classes, list) or not all(isinstance(c, (str, int)) for c in classes):
            raise MiroError(f"{path.name}: {key}: classes must be a list of names")
        confidence = str(item.get("confidence") or "normal")
        if confidence not in ("normal", "low"):
            raise MiroError(f"{path.name}: {key}: confidence must be normal or low")
        result[str(key)] = Suggestion(str(item.get("heading") or ""), [str(c) for c in classes], confidence,
                                      str(item.get("note") or ""))
    return result


def build_plan(miro_dir: Path, files: Mapping[str, str], suggestions_path: Optional[Path],
               class_names: Mapping[str, Sequence[str]]) -> MiroPlan:
    """files: Catalog model key -> PDF name in miro_dir. class_names: model key -> classes labelers draw."""
    models = {m.lower(): m for m in class_names}
    pages: List[MiroPage] = []
    for model, name in sorted(files.items()):
        key = models.get(str(model).lower())
        if key is None:
            raise MiroError(f"Miro file {name} names model {model}, which is not in the Catalog")
        pages.extend(extract_pdf(miro_dir / name, key))
    suggestions = read_suggestions(suggestions_path) if suggestions_path else {}
    by_key = {p.key: p for p in pages}
    problems = []
    for key, s in suggestions.items():
        page = by_key.get(key)
        if page is None:
            problems.append(f"{key}: no such page")
            continue
        unknown = [c for c in s.classes if c not in class_names[page.model]]
        if unknown:
            problems.append(f"{key}: not a {page.model} class: {', '.join(unknown)}")
    if problems:
        raise MiroError("Miro suggestions do not match: " + "; ".join(problems))
    report = {"pages": len(pages), "per_model": {m: sum(p.model == m for p in pages) for m in sorted({p.model for p in pages})},
              "suggested": sum(1 for s in suggestions.values() if s.classes),
              "without_page_suggestion": sorted(k for k in by_key if k not in suggestions)}
    return MiroPlan(pages, suggestions, report)
