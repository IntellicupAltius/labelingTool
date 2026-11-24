"""Releases: the Catalog's state turned into the files the rest of the system reads.

Pure functions only (no database, no disk writes), so every rule is testable on its own:
  - generate():   Catalog state -> the five outputs (pipeline POS map, blacklist, tool class lists,
                  training unite/exclude, documented classes);
  - parse_files() and read_production(): the same outputs read back into comparable content;
  - describe_changes(): plain-language difference between two contents;
  - parity():     generated content against today's production files, every difference explained or not;
  - run_checks(): the three checks that block a release.

The generated POS map follows mapping/generate_reconcile_map.py exactly (map order, first key wins,
skip entries excluded, classes in map order), so a release with no new decisions gives the same map.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml

from web_labeler.catalog.assortment import normalize_name as norm

HEADER = "# GENERISANO IZ KATALOGA. Ne menjati ručno. Broj izdanja i datum su u RELEASE.json.\n"
MAP_FILE = "pos_to_internal_mapping.yaml"
BLACKLIST_FILE = "blacklist_items.txt"
CLASS_LIST_DIR = "class_lists"
TRAINING_DIR = "training_classes"
DOCUMENTED_DIR = "documented_classes"
DECISIONS_SECTION = "# Katalog: ne pratimo (ručne odluke)"
ASSORTMENT_SECTION = "# Katalog: izbačeno iz asortimana"
_YAML_SPECIAL = ":{}[]|>&*!,'\""


@dataclass
class MapSource:
    status: str
    names: List[str]
    links: List[Tuple[str, str]]


@dataclass
class VariationState:
    """What the Catalog says about one POS name after the decisions in force."""
    identity_id: int
    article_id: int
    code: Optional[str]
    name: str
    state: str                      # tracked | ignored | pending | dormant
    links: List[Tuple[str, str]]
    decisions: List[Dict[str, Any]]  # decisions in force that set this (classes or ignore); empty = sources only
    ignore_why: Optional[str] = None   # "assortment" = left the menu; None = plain X


@dataclass
class ReleaseSources:
    entries: List[MapSource]
    blacklist_lines: List[str]
    variations: List[VariationState]
    classes: Dict[str, List[str]]            # tool model name (e.g. Glasses) -> names by class number
    training: Dict[str, Dict[str, Any]]      # model (lower) -> `classes:` section of the training config


@dataclass
class Generated:
    files: Dict[str, str]
    changes: List[Dict[str, Any]] = field(default_factory=list)    # what decisions changed against the sources
    conflicts: List[str] = field(default_factory=list)


# ---------------- Generating ----------------

def _map_text(mapping: Mapping[str, List[str]]) -> str:
    lines = [HEADER.rstrip("\n"), ""]
    for key in sorted(mapping):
        safe = f'"{key}"' if any(c in key for c in _YAML_SPECIAL) else key
        lines.append(f"{safe}:")
        lines.extend(f"- {cls}" for cls in mapping[key])
        lines.append("")
    return "\n".join(lines)


def _yaml_text(data: Any) -> str:
    return HEADER + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1000)


def _base_map(entries: Sequence[MapSource]) -> Tuple[Dict[str, List[str]], List[str]]:
    mapping: Dict[str, List[str]] = {}
    skip: List[str] = []
    for e in entries:
        if e.status == "skip":
            skip.extend(e.names)
            continue
        classes = [f"{m}.{c}" for m, c in e.links]
        if not classes:
            continue
        for key in e.names:
            if key in mapping and mapping[key] != classes:
                continue
            mapping[key] = classes
    return mapping, skip


def _is_comment(line: str) -> bool:
    return not line.strip() or line.strip().startswith("#")


def _expanded_exclude(names: Sequence[str], cfg: Mapping[str, Any]) -> List[str]:
    explicit = [str(n) for n in cfg.get("exclude") or []]
    prefixes = [str(p) for p in cfg.get("exclude_prefixes") or []]
    return sorted(set(explicit) | {n for n in names if any(n.startswith(p) for p in prefixes)})


def generate(src: ReleaseSources) -> Generated:
    mapping, skip_keys = _base_map(src.entries)
    blacklist = list(src.blacklist_lines)
    listed = {norm(l) for l in blacklist if not _is_comment(l)}
    extra_ignored = [k for k in dict.fromkeys(skip_keys) if norm(k) not in listed]
    left_menu: List[str] = []
    out = Generated({})

    by_name: Dict[str, List[VariationState]] = {}
    for v in src.variations:
        if v.decisions:
            by_name.setdefault(norm(v.name), []).append(v)
    for key, group in sorted(by_name.items()):
        wanted = {("ignore",) if v.state == "ignored" else tuple(f"{m}.{c}" for m, c in v.links) for v in group}
        if len(wanted) > 1:
            out.conflicts.append(f"{group[0].name.strip()}: isto ime ima različite odluke ({len(wanted)} različitih).")
            continue
        want = wanted.pop()
        map_keys = [k for k in mapping if norm(k) == key]
        before = ("ignore",) if key in listed or key in {norm(k) for k in extra_ignored} else (
            tuple(mapping[map_keys[0]]) if map_keys else ())
        if want == before:
            continue
        display = map_keys[0] if map_keys else group[0].name.strip()
        for k in map_keys:
            del mapping[k]
        blacklist = [l for l in blacklist if _is_comment(l) or norm(l) != key]
        extra_ignored = [k for k in extra_ignored if norm(k) != key]
        if want == ("ignore",):
            (left_menu if any(v.ignore_why == "assortment" for v in group) else extra_ignored).append(group[0].name.strip())
        elif want:
            mapping[display] = list(want)
        decisions = [d for v in group for d in v.decisions]
        out.changes.append({"name": display, "key": key, "before": list(before), "after": list(want),
                            "decisions": sorted({d["id"]: d for d in decisions}.values(), key=lambda d: d["id"])})
    if extra_ignored:
        blacklist += ["", DECISIONS_SECTION] + extra_ignored
    if left_menu:
        blacklist += ["", ASSORTMENT_SECTION] + left_menu

    files = {MAP_FILE: _map_text(mapping), BLACKLIST_FILE: HEADER + "\n".join(blacklist) + "\n"}
    for model, names in sorted(src.classes.items()):
        files[f"{CLASS_LIST_DIR}/{model}.yaml"] = _yaml_text({"nc": len(names), "names": list(names)})
        key = model.lower()
        cfg = src.training.get(key)
        if cfg is not None:
            files[f"{TRAINING_DIR}/{key}.yaml"] = _yaml_text({"classes": cfg})
        cfg = cfg or {}
        unite = [{"into": str(u.get("as") or u["classes"][0]), "classes": [str(c) for c in u["classes"]]}
                 for u in cfg.get("unite") or [] if u.get("classes")]
        files[f"{DOCUMENTED_DIR}/{key}.yaml"] = _yaml_text({
            "model": key, "source": "Katalog", "classes": list(names),
            "excluded": [{"class": n} for n in _expanded_exclude(names, cfg)], "unite": unite})
    out.files = files
    return out


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------- Reading outputs back ----------------

@dataclass
class Content:
    mapping: Dict[str, List[str]]                 # exact key -> classes
    blacklist: Dict[str, str]                     # normalized -> as written
    class_lists: Dict[str, List[str]]             # tool model name -> names
    training: Dict[str, Dict[str, Any]]           # model -> normalized unite/exclude
    documented: Dict[str, Dict[str, Any]]         # model -> normalized documented classes
    missing: List[str] = field(default_factory=list)


def _blacklist(text: str) -> Dict[str, str]:
    return {norm(l): l.strip() for l in text.splitlines() if not _is_comment(l)}


def _norm_training(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = cfg or {}
    return {"exclude": sorted(str(x) for x in cfg.get("exclude") or []),
            "exclude_prefixes": sorted(str(x) for x in cfg.get("exclude_prefixes") or []),
            "unite": sorted([str(u.get("as") or u["classes"][0]), sorted(str(c) for c in u["classes"])]
                            for u in cfg.get("unite") or [] if isinstance(u, Mapping) and u.get("classes"))}


def _norm_documented(doc: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    doc = doc or {}
    excluded = []
    for item in doc.get("excluded") or []:
        excluded.append(str(item.get("class")) if isinstance(item, Mapping) else str(item))
    return {"as_of": str(doc.get("documented_as_of") or "") or None,
            "classes": [str(c) for c in doc.get("classes") or []], "excluded": sorted(excluded),
            "unite": sorted([str(u.get("into") or u["classes"][0]), sorted(str(c) for c in u["classes"])]
                            for u in doc.get("unite") or [] if isinstance(u, Mapping) and u.get("classes"))}


def _yaml(text: str) -> Any:
    return yaml.safe_load(text)


def parse_files(files: Mapping[str, str]) -> Content:
    mapping = {str(k): [str(c) for c in v] for k, v in (_yaml(files.get(MAP_FILE, "")) or {}).items()}
    content = Content(mapping, _blacklist(files.get(BLACKLIST_FILE, "")), {}, {}, {})
    for path, text in files.items():
        folder, _, name = path.partition("/")
        stem = name[:-5] if name.endswith(".yaml") else name
        if folder == CLASS_LIST_DIR:
            content.class_lists[stem] = [str(n) for n in (_yaml(text) or {}).get("names") or []]
        elif folder == TRAINING_DIR:
            content.training[stem] = _norm_training((_yaml(text) or {}).get("classes"))
        elif folder == DOCUMENTED_DIR:
            content.documented[stem] = _norm_documented(_yaml(text))
    return content


@dataclass(frozen=True)
class ProductionPaths:
    pipeline_map: Optional[Path]
    blacklist: Optional[Path]
    class_lists_dir: Optional[Path]
    training_dir: Optional[Path]
    documented_dir: Optional[Path]


def _read(path: Optional[Path]) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8") if path is not None and path.is_file() else None
    except OSError:
        return None


def read_production(paths: ProductionPaths, models: Iterable[str]) -> Content:
    """Today's files, read only. A file that is missing is listed, not guessed."""
    missing: List[str] = []
    map_text = _read(paths.pipeline_map)
    if map_text is None:
        missing.append("mapa za pipeline")
    black_text = _read(paths.blacklist)
    if black_text is None:
        missing.append("blacklist")
    content = Content({str(k): [str(c) for c in v] for k, v in (_yaml(map_text or "") or {}).items()},
                      _blacklist(black_text or ""), {}, {}, {}, missing)
    for model in models:
        key = model.lower()
        text = None
        if paths.class_lists_dir is not None:
            text = _read(paths.class_lists_dir / f"{model}.yaml") or _read(paths.class_lists_dir / f"{model}.yml")
        if text is None:
            missing.append(f"lista klasa {model}")
        else:
            content.class_lists[model] = [str(n) for n in ((_yaml(text) or {}).get("names") or [])]
        cfg = _yaml(_read(paths.training_dir / f"{key}.yaml") or "") if paths.training_dir else None
        if isinstance(cfg, Mapping) and isinstance(cfg.get("classes"), Mapping):
            content.training[key] = _norm_training(cfg["classes"])
        doc = _yaml(_read(paths.documented_dir / f"{key}.yaml") or "") if paths.documented_dir else None
        if isinstance(doc, Mapping):
            content.documented[key] = _norm_documented(doc)
    return content


# ---------------- Plain-language differences ----------------

def _classes_text(classes: Sequence[str]) -> str:
    return ", ".join(classes)


def describe_changes(old: Content, new: Content) -> List[Dict[str, Any]]:
    """What a release changes against the one before it (or today's files), one line per thing."""
    out: List[Dict[str, Any]] = []
    old_map = {norm(k): (k, v) for k, v in old.mapping.items()}
    new_map = {norm(k): (k, v) for k, v in new.mapping.items()}
    for key in sorted(set(old_map) | set(new_map)):
        before, after = old_map.get(key), new_map.get(key)
        name = (after or before)[0]
        if before and after:
            if before[1] != after[1]:
                out.append({"area": "map", "key": key, "kind": "changed",
                            "text": f"{name}: klasa se menja sa {_classes_text(before[1])} na {_classes_text(after[1])}."})
        elif after:
            how = "više nije na listi za ignorisanje i " if key in old.blacklist else ""
            out.append({"area": "map", "key": key, "kind": "added",
                        "text": f"{name}: {how}počinje da se prati kao {_classes_text(after[1])}."})
        elif key in new.blacklist:
            out.append({"area": "map", "key": key, "kind": "ignored", "text": f"{name}: više se ne prati (ide na listu za ignorisanje)."})
        else:
            out.append({"area": "map", "key": key, "kind": "lost", "text": f"{name}: nestaje iz mape, a ne ide ni na listu za ignorisanje."})
    for key in sorted(set(new.blacklist) - set(old.blacklist) - set(old_map)):
        out.append({"area": "blacklist", "key": key, "kind": "ignored", "text": f"{new.blacklist[key]}: dodaje se na listu za ignorisanje."})
    for key in sorted(set(old.blacklist) - set(new.blacklist) - set(new_map)):
        out.append({"area": "blacklist", "key": key, "kind": "unignored",
                    "text": f"{old.blacklist[key]}: skida se sa liste za ignorisanje, a nema klasu (čekaće odluku)."})
    for model in sorted(set(old.class_lists) | set(new.class_lists)):
        a, b = old.class_lists.get(model, []), new.class_lists.get(model, [])
        if a != b:
            added = [n for n in b if n not in a]
            text = f"{model}: nove klase {', '.join(added)}." if added and b[:len(a)] == a else f"{model}: lista klasa se menja."
            out.append({"area": "class_lists", "key": model, "kind": "changed", "text": text})
    for area, label in (("training", "spajanja i izbacivanja za trening"), ("documented", "dokumentovane klase")):
        old_part, new_part = getattr(old, area), getattr(new, area)
        for model in sorted(set(old_part) | set(new_part)):
            if old_part.get(model) != new_part.get(model):
                out.append({"area": area, "key": model, "kind": "changed", "text": f"{model}: menjaju se {label}."})
    return out


# ---------------- Parity with today's files ----------------

AREA_LABELS = {"map": "Mapa za pipeline", "blacklist": "Lista za ignorisanje", "class_lists": "Liste klasa",
               "training": "Spajanja i izbacivanja", "documented": "Dokumentovane klase"}


def _short(items: Sequence[Any], limit: int = 8) -> str:
    shown = [" + ".join(i[1]) + " -> " + i[0] if isinstance(i, list) else str(i) for i in items[:limit]]
    return ", ".join(shown) + (f" i još {len(items) - limit}" if len(items) > limit else "")


def _doc_details(doc: Mapping[str, Any], gen: Mapping[str, Any]) -> List[str]:
    out = []
    labels = {"classes": "Klase", "excluded": "Izbačene klase", "unite": "Spajanja"}
    for key, label in labels.items():
        a, b = doc.get(key) or [], gen.get(key) or []
        if a == b:
            continue
        if key == "classes" and not a:
            out.append(f"{label}: dokument nema spisak klasa (Katalog ima {len(b)}).")
            continue
        only_doc, only_cat = [x for x in a if x not in b], [x for x in b if x not in a]
        parts = []
        if only_cat:
            parts.append(f"u dokumentu nema {_short(only_cat)}")
        if only_doc:
            parts.append(f"samo dokument ima {_short(only_doc)}")
        out.append(f"{label}: " + ("; ".join(parts) if parts else "isti spisak, drugi redosled") + ".")
    return out


def _decision_text(decisions: Sequence[Mapping[str, Any]]) -> str:
    parts = [f"odluka #{d['id']} ({d['actor']}, {str(d['created_at'])[:10]}){': ' + d['reason'] if d.get('reason') else ''}"
             for d in decisions]
    return "Ručna odluka u Katalogu: " + "; ".join(parts) + "."


def parity(production: Content, generated: Generated, new: Content) -> Dict[str, Any]:
    """Every difference between the generated outputs and today's files, each explained or marked unexplained."""
    by_key = {c["key"]: c for c in generated.changes}
    items: List[Dict[str, Any]] = []
    for change in describe_changes(production, new):
        item = {"area": change["area"], "area_label": AREA_LABELS[change["area"]], "text": change["text"], "explained": False,
                "why": "Nije objašnjeno: razlika ne dolazi ni od jedne odluke u Katalogu."}
        if change["area"] in ("map", "blacklist") and change["key"] in by_key:
            item.update(explained=True, why=_decision_text(by_key[change["key"]]["decisions"]))
        elif change["area"] == "documented":
            model = change["key"]
            gen_doc = new.documented.get(model, {})
            prod_training = production.training.get(model)
            expected = {"excluded": gen_doc.get("excluded"), "unite": gen_doc.get("unite")}
            agrees = prod_training is not None and new.training.get(model) == prod_training
            prod_doc = production.documented.get(model, {})
            details = _doc_details(prod_doc, gen_doc)
            item["details"] = details
            if agrees and details:
                when = f" (zapisan {prod_doc['as_of']})" if prod_doc.get("as_of") else ""
                item.update(explained=True, why=(
                    f"Današnji fajl dokumentovanih klasa{when} zaostaje za trening konfiguracijom i listama klasa alata. "
                    "Katalog prati njih, kao i trening. Ista razlika je u pogledu Razlike izvora."))
        items.append(item)
    for name in production.missing:
        items.append({"area": "missing", "area_label": "Nedostaje", "text": f"Današnji fajl nije pronađen: {name}.",
                      "explained": False, "why": "Ne može da se uporedi."})
    counts = {area: sum(1 for i in items if i["area"] == area) for area in AREA_LABELS}
    return {"identical": not items, "unexplained": sum(1 for i in items if not i["explained"]),
            "counts": counts, "items": items}


# ---------------- Checks that block a release ----------------

def run_checks(new: Content, baseline: Content, generated: Optional[Generated], sources: Optional[ReleaseSources],
               live_classes: Mapping[str, Optional[Set[str]]]) -> List[Dict[str, Any]]:
    """live_classes: model (lower) -> class names of the live model, None when that list cannot be read."""
    missing: List[str] = []
    for key in sorted(new.mapping):
        for item in new.mapping[key]:
            model, _, cls = item.partition(".")
            live = live_classes.get(model.lower())
            if live is None:
                missing.append(f"{key}: traži {item}, a lista klasa živog modela {model} ne može da se pročita.")
            elif cls not in live:
                missing.append(f"{key}: traži {item}, a te klase nema u živom modelu.")

    explained = {c["key"] for c in generated.changes} if generated else set()
    new_keys = {norm(k) for k in new.mapping} | set(new.blacklist)
    lost: List[str] = []
    for key in sorted(baseline.mapping):
        if norm(key) not in new_keys and norm(key) not in explained:
            lost.append(f"{key}: bio je u mapi, a u novom izdanju nije ni u mapi ni na listi za ignorisanje.")
    if sources is not None:
        for v in sources.variations:
            if v.state == "tracked" and v.links and norm(v.name) not in {norm(k) for k in new.mapping}:
                lost.append(f"{v.name.strip()}: Katalog ga prati, a u mapi ga nema.")

    duplicates: List[str] = list(generated.conflicts) if generated else []
    if sources is not None:
        by_code: Dict[str, Dict[int, List[str]]] = {}
        for v in sources.variations:
            if v.code and v.state != "ignored":
                by_code.setdefault(v.code, {}).setdefault(v.article_id, []).append(v.name.strip())
        for code, articles in sorted(by_code.items()):
            if len(articles) > 1:
                names = "; ".join(", ".join(n) for n in articles.values())
                duplicates.append(f"Šifra {code} pripada različitim artiklima: {names}.")
    return [
        {"check": "classes", "label": "Svaka klasa postoji u živom modelu", "ok": not missing, "problems": missing},
        {"check": "lost", "label": "Nijedan artikal se ne gubi", "ok": not lost, "problems": lost},
        {"check": "duplicates", "label": "Nema duplih šifri ni sukoba odluka", "ok": not duplicates, "problems": duplicates},
    ]


def release_json(number: int, kind: str, created_at: str, actor: str, reason: str, based_on: Optional[int],
                 checksums: Mapping[str, str]) -> str:
    return json.dumps({"release": number, "kind": kind, "created_at": created_at, "actor": actor, "reason": reason,
                       "based_on": based_on, "note": "Generisano iz Kataloga. Ne menjati ručno.",
                       "files": dict(sorted(checksums.items()))}, ensure_ascii=False, indent=2) + "\n"
