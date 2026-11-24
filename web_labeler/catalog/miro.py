"""Old Miro boards: every PDF page becomes one whole image. Pages are never cut into articles.

Only reads the PDFs. Extraction runs in a temporary folder; the Catalog stores the result itself.

A new export from Miro re-encodes every image and may reorder the pages, so a page is recognised by how
it looks (a small grey thumbnail and its shape), never by its bytes or its position in the PDF.
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

_PAGE_KEY = re.compile(r"^[a-z0-9_]+-[0-9a-f]{2,10}$")

# Mean difference of the 24x24 grey thumbnails (0-255) below which two pages are the same page.
# Measured on two Bottles exports (09.10 and 24.11.2026): unchanged pages 0.3-0.9, changed pages 11.7 and more.
SAME_PAGE_LIMIT = 5.0
# Pages whose height/width ratios differ by more than this are never the same page.
SAME_SHAPE_LIMIT = 0.06
_THUMB = 24


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
    signature: bytes = b""


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
    """Key of a page in the first import of a model: its position."""
    return f"{model.lower()}-{page_no:02d}"


def content_key(model: str, sha256: str) -> str:
    """Key of a page that appears in a later export: stable for that image wherever it sits in the PDF."""
    return f"{model.lower()}-{sha256[:10]}"


def key_model(key: str) -> str:
    return key.rsplit("-", 1)[0]


def signature(data: bytes) -> bytes:
    from io import BytesIO
    from PIL import Image
    with Image.open(BytesIO(data)) as im:
        return im.convert("L").resize((_THUMB, _THUMB)).tobytes()


def page_distance(sig_a: bytes, size_a: Tuple[int, int], sig_b: bytes, size_b: Tuple[int, int]) -> Optional[float]:
    """How different two pages look; None when they cannot be the same page."""
    if not sig_a or len(sig_a) != len(sig_b) or min(*size_a, *size_b) <= 0:
        return None
    ratio_a, ratio_b = size_a[1] / size_a[0], size_b[1] / size_b[0]
    if abs(ratio_a - ratio_b) / max(ratio_a, ratio_b) > SAME_SHAPE_LIMIT:
        return None
    return sum(abs(a - b) for a, b in zip(sig_a, sig_b)) / len(sig_a)


def match_pages(new: Sequence[Tuple[bytes, Tuple[int, int]]], old: Sequence[Tuple[bytes, Tuple[int, int]]]) -> Dict[int, int]:
    """One-to-one: index in new -> index in old, closest pairs first, only pairs within SAME_PAGE_LIMIT."""
    pairs = []
    for i, (sig_n, size_n) in enumerate(new):
        for j, (sig_o, size_o) in enumerate(old):
            d = page_distance(sig_n, size_n, sig_o, size_o)
            if d is not None and d <= SAME_PAGE_LIMIT:
                pairs.append((d, i, j))
    taken_new, taken_old, result = set(), set(), {}
    for _, i, j in sorted(pairs):
        if i not in taken_new and j not in taken_old:
            result[i] = j
            taken_new.add(i)
            taken_old.add(j)
    return result


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
                                   hashlib.sha256(data).hexdigest(), width, height, signature(data)))
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
    # A suggestion is keyed by the page's content key, or by the key the page already has in the Catalog
    # (which pages those are is only known when the import meets the database).
    problems = []
    for key, s in suggestions.items():
        model = models.get(key_model(key))
        if model is None:
            problems.append(f"{key}: no such model")
            continue
        unknown = [c for c in s.classes if c not in class_names[model]]
        if unknown:
            problems.append(f"{key}: not a {model} class: {', '.join(unknown)}")
    if problems:
        raise MiroError("Miro suggestions do not match: " + "; ".join(problems))
    report = {"pages": len(pages), "per_model": {m: sum(p.model == m for p in pages) for m in sorted({p.model for p in pages})}}
    return MiroPlan(pages, suggestions, report)
