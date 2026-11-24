"""Read training config, documented classes, cards, Word exports and live class lists. Never writes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import yaml

TRAINED, EXCLUDED, NOT_IN_MODEL, UNKNOWN = "trained", "excluded", "not_in_model", "unknown"
HIGH, MEDIUM, INFO = "high", "medium", "info"
SRC_CONFIG = "Trening konfiguracija"
SRC_DOCUMENTED = "Dokumentovane klase (odraz Word-a i Mira)"
SRC_LIVE = "Živi model"
SRC_CARD = "Kartica"
SRC_TOOL = "Lista klasa alata"
SRC_MAP = "POS mapa"

_CLASS_NAME = re.compile(r"^[A-Z0-9][A-Z0-9_\-]*$")
_INTERNAL_NOTE = re.compile(r"\s*\([^()]*(?:DATA-\d|MDQ-\d|LR-\d|GT \d|\bO\d+[ab]?\b)[^()]*\)")
_CARD_FIELDS = {"prepoznaje se": "recognize", "primeri": "examples", "meša se sa": "confused", "ne": "note"}


@dataclass(frozen=True)
class RuleSources:
    training_config_dir: Path
    documented_dir: Optional[Path] = None
    cards_dir: Optional[Path] = None
    word_snapshots_dir: Optional[Path] = None
    live_models_dir: Optional[Path] = None
    rule_files: Mapping[str, Tuple[str, Path]] = field(default_factory=dict)
    known_conflicts: Optional[Path] = None


@dataclass
class RulesPlan:
    classes: Dict[str, Dict[str, Dict[str, Any]]]
    texts: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    report: Dict[str, Any]
    training_classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # model -> `classes:` of the training config


def _read_yaml(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _clean(text: str) -> str:
    text = re.sub(r"[*`]+", "", text).replace("⚠️", "").strip()
    text = _INTERNAL_NOTE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_card_file(text: str) -> Tuple[Dict[str, List[Tuple[str, str]]], bool]:
    """Return {class: [(kind, text)]} and whether the file is marked as a draft."""
    cards: Dict[str, List[Tuple[str, str]]] = {}
    current: Optional[str] = None
    bullet: Optional[List[str]] = None

    def flush() -> None:
        nonlocal bullet
        if current is None or bullet is None:
            bullet = None
            return
        raw = " ".join(bullet)
        bullet = None
        match = re.match(r"^\*\*([^*:]+?)(?:\s*\([^)]*\))?:\*\*\s*(.*)$", raw)
        label = match.group(1).strip().lower() if match else ""
        body = _clean(match.group(2) if match else raw)
        if label == "slika" or raw.startswith("Slika:") or not body:
            return
        cards[current].append((_CARD_FIELDS.get(label, "note"), body))

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if heading:
            flush()
            name = heading.group(2)
            current = name if len(heading.group(1)) == 3 and _CLASS_NAME.match(name) else None
            if current:
                cards.setdefault(current, [])
        elif line.startswith("- "):
            flush()
            bullet = [line[2:].strip()]
        elif bullet is not None and line.startswith((" ", "\t")) and line.strip():
            bullet.append(line.strip())
        else:
            flush()
    flush()
    return cards, "NACRT" in "\n".join(text.splitlines()[:12])


def parse_rule_file(text: str, known: Set[str]) -> Dict[str, List[str]]:
    """Newer per-class rules: table rows, and bullets that name a class exactly."""
    rules: Dict[str, List[str]] = {}
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            for name in re.split(r"\s*/\s*", cells[0].replace("*", "")):
                if name in known:
                    rules.setdefault(name, []).append(_clean(cells[1]))
        elif line.startswith("- ") and "RELABEL" not in line:
            names = [n for n in re.findall(r"[A-Z0-9][A-Z0-9_\-]+", line) if n in known]
            for name in dict.fromkeys(names):
                rules.setdefault(name, []).append(_clean(line[2:]))
    return rules


def resolve_training(names: List[str], config: Mapping[str, Any]) -> Tuple[Set[str], Dict[str, str], List[Dict[str, Any]]]:
    """Same reading as the training builder: exclude, exclude prefixes, then unite."""
    classes = config.get("classes") or {}
    excluded = set(classes.get("exclude") or [])
    for prefix in classes.get("exclude_prefixes") or []:
        excluded |= {n for n in names if n.startswith(prefix)}
    groups: List[Dict[str, Any]] = []
    target: Dict[str, str] = {}
    for group in classes.get("unite") or []:
        members = list(group.get("classes") or [])
        if len(members) < 2:
            continue
        canonical = group.get("as") or members[0]
        groups.append({"as": canonical, "classes": members})
        for member in members:
            target[member] = canonical
    return excluded, target, groups


def _newest_word_export(root: Optional[Path], model: str) -> Tuple[Optional[str], Optional[str]]:
    if root is None or not root.is_dir():
        return None, None
    candidates = sorted(root.glob(f"*/text/overview_{model}.txt"), key=lambda p: p.parts[-3])
    if not candidates:
        return None, None
    path = candidates[-1]
    try:
        return path.parts[-3], path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, None


def _date_label(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d.%m.%Y.")


def _finding(model: Optional[str], class_name: Optional[str], severity: str, title: str, text: str,
             sources: List[Tuple[str, str]]) -> Dict[str, Any]:
    return {"model": model, "class": class_name, "severity": severity, "title": title, "text": text,
            "sources": [{"label": label, "says": says} for label, says in sources]}


def _join(items) -> str:
    return ", ".join(sorted(items))


def _compare_documented(model: str, names: List[str], excluded: Set[str], groups: List[Dict[str, Any]],
                        documented: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    doc_excluded: Dict[str, str] = {}
    for item in documented.get("excluded") or []:
        name = item.get("class") if isinstance(item, dict) else item
        if name:
            doc_excluded[str(name)] = str(item.get("reason", "")) if isinstance(item, dict) else ""
    moved: Dict[str, str] = {}
    for key in ("merged_per_doc", "remapped_per_doc"):
        for item in documented.get(key) or []:
            moved[str(item.get("class"))] = str(item.get("into") or item.get("doc") or "")
    united = {m for g in groups for m in g["classes"]}
    doc_groups = [{"as": str(g.get("into") or ""), "classes": [str(c) for c in g.get("classes") or []]}
                  for g in documented.get("unite") or []]
    doc_united = {m for g in doc_groups for m in g["classes"]}

    for name in sorted(excluded - set(doc_excluded)):
        if name in moved:
            out.append(_finding(model, name, INFO, f"{name}: izbačena ili prebačena?",
                                "Trening ovu klasu izbacuje, a dokumentacija kaže da je prebačena u drugu klasu.",
                                [(SRC_CONFIG, "izbačena iz treninga"), (SRC_DOCUMENTED, f"prebačena: {moved[name]}")]))
        else:
            out.append(_finding(model, name, MEDIUM, f"{name}: izbačena, a dokumentacija to ne kaže",
                                "Trening ovu klasu izbacuje, a u dokumentaciji nije navedena kao izbačena.",
                                [(SRC_CONFIG, "izbačena iz treninga"), (SRC_DOCUMENTED, "nije među izbačenim")]))
    for name in sorted(set(doc_excluded) - excluded):
        if name in united and name in doc_united:
            continue  # both sides retire the class by merging it; the merge itself is compared below
        how = "spaja se u treningu" if name in united else "ulazi u trening"
        out.append(_finding(model, name, MEDIUM, f"{name}: dokumentacija je izbacuje, trening ne",
                            f"Dokumentacija kaže da je klasa izbačena, a u treningu {how}.",
                            [(SRC_DOCUMENTED, doc_excluded[name] or "izbačena"), (SRC_CONFIG, how)]))
    for group in groups:
        members = set(group["classes"])
        same = [d for d in doc_groups if set(d["classes"]) == members]
        if same and same[0]["as"] == group["as"]:
            continue
        near = [d for d in doc_groups if set(d["classes"]) & members]
        said = "; ".join(f"{_join(d['classes'])} u {d['as']}" for d in near) or "ovo spajanje nije zapisano"
        out.append(_finding(model, group["as"], MEDIUM, f"Spajanje u {group['as']} nije isto zapisano",
                            "Trening spaja ove klase drugačije nego što piše u dokumentaciji.",
                            [(SRC_CONFIG, f"{_join(members)} u {group['as']}"), (SRC_DOCUMENTED, said)]))
    config_members = [set(g["classes"]) for g in groups]
    for group in doc_groups:
        if not any(set(group["classes"]) & members for members in config_members):
            out.append(_finding(model, group["as"], MEDIUM, f"Spajanje u {group['as']} nije u treningu",
                                "Dokumentacija navodi spajanje koje trening ne radi.",
                                [(SRC_DOCUMENTED, f"{_join(group['classes'])} u {group['as']}"),
                                 (SRC_CONFIG, "ne spaja ove klase")]))
    listed = documented.get("classes")
    if isinstance(listed, list):
        missing, extra = set(names) - set(map(str, listed)), set(map(str, listed)) - set(names)
        if missing or extra:
            out.append(_finding(model, None, MEDIUM, "Lista klasa u dokumentaciji nije ista kao u alatu",
                                "Dokumentacija i alat ne navode iste klase.",
                                [(SRC_TOOL, f"nema u dokumentaciji: {_join(missing) or 'ništa'}"),
                                 (SRC_DOCUMENTED, f"nema u alatu: {_join(extra) or 'ništa'}")]))
    return out


def _known_conflicts(path: Optional[Path]) -> List[Dict[str, Any]]:
    data = _read_yaml(path)
    out: List[Dict[str, Any]] = []
    for item in (data or {}).get("conflicts") or []:
        changed: List[str] = []
        sources: List[Tuple[str, str]] = []
        for source in item.get("sources") or []:
            sources.append((str(source["label"]), str(source["says"])))
            if source.get("file"):
                try:
                    content = Path(str(source["file"])).expanduser().read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    content = ""
                if str(source["phrase"]) not in content:
                    changed.append(str(source["label"]))
        text = str(item["text"])
        if changed:
            text += " Proveriti: tekst se promenio u izvoru: " + ", ".join(changed) + "."
        out.append(_finding(str(item["model"]), str(item["class"]), HIGH, str(item["title"]), text, sources))
    return out


def build_rules_plan(sources: RuleSources, class_lists: Mapping[str, List[str]],
                     map_classes: Mapping[str, Mapping[str, bool]]) -> RulesPlan:
    """map_classes: {model: {class: is any article of that class actually sold}}."""
    classes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    texts: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    training_classes: Dict[str, Dict[str, Any]] = {}
    for model_name, names in sorted(class_lists.items()):
        model = model_name.lower()
        config = _read_yaml(sources.training_config_dir / f"{model}.yaml")
        if config is None:
            findings.append(_finding(model, None, HIGH, "Nema trening konfiguracije",
                                     "Za ovaj model ne može da se utvrdi šta se spaja i izbacuje.",
                                     [(SRC_CONFIG, "fajl nije pronađen ili nije ispravan")]))
            config = {}
        excluded, target, groups = resolve_training(names, config)
        if isinstance(config.get("classes"), dict):
            training_classes[model] = {k: config["classes"][k] for k in ("exclude", "exclude_prefixes", "unite")
                                       if k in config["classes"]}
        live_data = _read_yaml(sources.live_models_dir / model / "active_classes.yaml") if sources.live_models_dir else None
        live = {str(n) for n in (live_data or {}).get("names") or [] if str(n) != "background"} or None

        model_rows: Dict[str, Dict[str, Any]] = {}
        for class_id, name in enumerate(names):
            trained = None if name in excluded else target.get(name, name)
            if name in excluded:
                state = EXCLUDED
            elif live is None:
                state = UNKNOWN
            elif trained in live:
                state = TRAINED
            else:
                state = NOT_IN_MODEL
                findings.append(_finding(model, name, MEDIUM, f"{name}: nije izbačena, a model je nema",
                                         "Trening ovu klasu ne izbacuje, ali je živi model ne zna. Najčešći razlog je da nema slika.",
                                         [(SRC_CONFIG, "klasa ulazi u trening"), (SRC_LIVE, "klase nema")]))
            model_rows[name] = {"class_id": class_id, "trained_name": trained, "state": state}
        classes[model_name] = model_rows

        if live is not None:
            expected = {r["trained_name"] for r in model_rows.values() if r["trained_name"]}
            extra = live - expected
            if extra:
                findings.append(_finding(model, None, HIGH, "Živi model ima klase koje trening ne objašnjava",
                                         f"Klase: {_join(extra)}.",
                                         [(SRC_LIVE, _join(extra)), (SRC_CONFIG, "ne proizvodi ove klase")]))
            used = map_classes.get(model, {})
            for name in sorted(set(used) - live):
                sold = used[name]
                findings.append(_finding(model, name, HIGH if sold else INFO,
                                         f"POS mapa traži klasu koju model nema: {name}",
                                         "Prodati artikli vezani za ovu klasu ne mogu da se upare preko nje." if sold
                                         else "Artikli vezani za ovu klasu nisu prodavani, pa to sada ne pravi štetu.",
                                         [(SRC_MAP, f"koristi {name}"), (SRC_LIVE, "klase nema")]))

        documented = _read_yaml(sources.documented_dir / f"{model}.yaml") if sources.documented_dir else None
        if documented is None:
            findings.append(_finding(model, None, INFO, "Nema dokumentovanih klasa",
                                     "Spajanja i izbacivanja nisu mogla da se uporede sa dokumentacijom.",
                                     [(SRC_DOCUMENTED, "fajl nije pronađen")]))
        else:
            findings.extend(_compare_documented(model, names, excluded, groups, documented))

        known = set(names) | {r["trained_name"] for r in model_rows.values() if r["trained_name"]}
        card_path = sources.cards_dir / f"kartice_{model}_v1.md" if sources.cards_dir else None
        cards: Dict[str, List[Tuple[str, str]]] = {}
        draft = False
        if card_path is not None and card_path.is_file():
            cards, draft = parse_card_file(card_path.read_text(encoding="utf-8"))
        source_label = f"{SRC_CARD} {model_name}" + (" (nacrt)" if draft else "")
        for name, items in cards.items():
            for position, (kind, text) in enumerate(items):
                texts.append({"model": model_name, "class": name, "position": 100 + position, "kind": kind,
                              "text": text, "source": source_label})
            if name not in known and not items:
                continue  # a section heading such as OSTALO, not a class card
            if name not in known:
                findings.append(_finding(model, name, INFO, f"Kartica za nepoznatu klasu: {name}",
                                         "Postoji kartica, a takve klase nema ni u alatu ni u modelu.",
                                         [(source_label, "ima karticu"), (SRC_TOOL, "klase nema")]))
            elif name in excluded:
                findings.append(_finding(model, name, INFO, f"{name}: ima karticu, a izbačena je iz treninga",
                                         "Labeleri mogu da je crtaju po kartici, ali model tu klasu ne uči.",
                                         [(source_label, "ima karticu"), (SRC_CONFIG, "izbačena iz treninga")]))
        for name, row in model_rows.items():
            if row["state"] != EXCLUDED and not cards.get(name) and not cards.get(row["trained_name"] or ""):
                findings.append(_finding(model, name, MEDIUM, f"{name}: nema kartice",
                                         "Klasa se koristi, a nema opis kako se prepoznaje.",
                                         [(source_label, "kartice nema"), (SRC_TOOL, "klasa postoji")]))

        if model in sources.rule_files:
            label, path = sources.rule_files[model]
            if path.is_file():
                for name, items in parse_rule_file(path.read_text(encoding="utf-8"), known).items():
                    for position, text in enumerate(items):
                        texts.append({"model": model_name, "class": name, "position": position, "kind": "rule",
                                      "text": text, "source": label})

        export, word = _newest_word_export(sources.word_snapshots_dir, model_name)
        if word is None:
            findings.append(_finding(model, None, INFO, "Nema izvoza Word dokumenta",
                                     "Word nije mogao da se uporedi.", [("Word", "izvoz nije pronađen")]))
        else:
            word_label = f"Word {model_name} (izvoz {_date_label(export)})"
            flat = word.upper().replace(" ", "_")
            for name, row in model_rows.items():
                if row["state"] != EXCLUDED and name.upper() not in flat:
                    findings.append(_finding(model, name, MEDIUM, f"{name}: Word je ne pominje",
                                             "Klasa se koristi, a u poslednjem izvozu Word-a je nema.",
                                             [(word_label, "ne pominje klasu"), (SRC_TOOL, "klasa postoji")]))
            config_path = sources.training_config_dir / f"{model}.yaml"
            if config_path.is_file():
                changed = datetime.fromtimestamp(config_path.stat().st_mtime).date()
                if changed.isoformat() > export:
                    findings.append(_finding(model, None, INFO, "Word je stariji od trening konfiguracije",
                                             "Spajanja i izbacivanja su možda menjana posle poslednjeg izvoza Word-a.",
                                             [(word_label, "poslednji izvoz"),
                                              (SRC_CONFIG, f"izmenjena {changed.strftime('%d.%m.%Y.')}")]))

    findings.extend(_known_conflicts(sources.known_conflicts))
    order = {HIGH: 0, MEDIUM: 1, INFO: 2}
    findings.sort(key=lambda f: (order[f["severity"]], f["model"] or "", f["class"] or "", f["title"]))
    counts = {k: sum(f["severity"] == k for f in findings) for k in order}
    return RulesPlan(classes, texts, findings, {"findings": counts, "rule_texts": len(texts)}, training_classes)
