#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# ⇩⇩⇩  TVOJ .txt PATH  ⇩⇩⇩
PATH = "/home/aleksanovevski/Documents/Projects/intellicupLabelingTool/Blaznavac Venue - Labeler Visual Analysis Template.txt"

# Hvatamo sve između uglastih zagrada kao "head[: tail]"
BRACKET_RE = re.compile(r'\[(?P<head>[^\]:\]]+)(?::\s*(?P<tail>[^\]]+))?\]', re.IGNORECASE)

# Ime posle art:/art_alt: — do " - [" ili "[" ili kraja reda
NAME_RE_ART     = re.compile(r'^\s*art:\s*(?P<name>.+?)(?:\s*(?=\-\s*\[|\[)|\s*$)', re.IGNORECASE)
NAME_RE_ART_ALT = re.compile(r'^\s*art_alt:\s*(?P<name>.+?)(?:\s*(?=\-\s*\[|\[)|\s*$)', re.IGNORECASE)

# Prepoznavanje isključivo kada linija POČINJE sa art:/art_alt:
STARTS_ART     = re.compile(r'^\s*art:\s', re.IGNORECASE)
STARTS_ART_ALT = re.compile(r'^\s*art_alt:\s', re.IGNORECASE)

def parse_model_class_from_tail(tail: str):
    """
    Iz 'tail' izvući model i klasu.
    Dozvoljeno: 'Model.Class' ili 'Model: Class'
    Model: [A-Za-z0-9_-]+
    Class: bilo šta do kraja
    """
    tail = tail.strip()
    m = re.match(r'^\s*([A-Za-z0-9_-]+)\s*(?:[.:])\s*(.+?)\s*$', tail)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Ako nema separatora, tretiraj sve kao class bez modela
    return None, tail.strip()

def main():
    with open(PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    total_art = 0
    total_art_alt = 0
    counts = {"INCL": 0, "MISS": 0, "NO_IMG": 0, "SKIP": 0, "PLAN": 0, "MISSING_TAG": 0, "MALFORMED_INCL": 0}

    details = []  # (seq_no, name, TAG, model, class, raw)
    seq_no = 0

    for raw in lines:
        line = raw.rstrip()

        # Samo linije koje KREĆU sa art:/art_alt:
        if STARTS_ART.match(line):
            m_name = NAME_RE_ART.search(line)
            total_art += 1
        elif STARTS_ART_ALT.match(line):
            m_name = NAME_RE_ART_ALT.search(line)
            total_art_alt += 1
        else:
            continue  # sve ostalo preskačemo (uklj. bullet objašnjenja)

        seq_no += 1  # redni broj u DETAILS

        name = (m_name.group("name").strip() if m_name else "")
        # ukloni eventualni završni '-' ako stoji sam
        name = re.sub(r'\s*-\s*$', '', name)

        m = BRACKET_RE.search(line)
        if not m:
            counts["MISSING_TAG"] += 1
            details.append((seq_no, name, "MISSING_TAG", "", "", line.strip()))
            continue

        head = (m.group("head") or "").strip()
        tail = (m.group("tail") or "").strip()
        head_upper = head.upper()

        if head_upper in ("MISS", "NO_IMG", "SKIP"):
            counts[head_upper] += 1
            details.append((seq_no, name, head_upper, "", "", line.strip()))

        elif head_upper == "PLAN":
            # opciono parsiraj predlog model/klase iz tail-a
            model, classname = ("", "")
            if tail:
                m, c = parse_model_class_from_tail(tail)
                model, classname = (m or "", c or "")
            counts["PLAN"] += 1
            details.append((seq_no, name, "PLAN", model, classname, line.strip()))

        elif head_upper == "INCL":
            if not tail:
                counts["MALFORMED_INCL"] += 1
                details.append((seq_no, name, "MALFORMED_INCL", "", "", line.strip()))
            else:
                model, classname = parse_model_class_from_tail(tail)
                if not model or not classname:
                    counts["MALFORMED_INCL"] += 1
                    details.append((seq_no, name, "MALFORMED_INCL", model or "", classname or "", line.strip()))
                else:
                    counts["INCL"] += 1
                    details.append((seq_no, name, "INCL", model, classname, line.strip()))
        else:
            # Implicitni INCL: [Model: Class] ili [Model.Class]
            if tail:
                model, classname = head, tail
            else:
                model, classname = parse_model_class_from_tail(head)

            if model and classname:
                counts["INCL"] += 1
                details.append((seq_no, name, "INCL", model, classname, line.strip()))
            else:
                counts["MISSING_TAG"] += 1
                details.append((seq_no, name, "MISSING_TAG", model or "", classname or "", line.strip()))

    # Sažetak
    total_all = total_art + total_art_alt
    print(f"Total 'art:' lines: {total_art}")
    print(f"Total 'art_alt:' lines: {total_art_alt}")
    print(f"Total items: {total_all}")
    print(f"  INCL: {counts['INCL']}")
    print(f"  MISS: {counts['MISS']}")
    print(f"  NO_IMG: {counts['NO_IMG']}")
    print(f"  SKIP: {counts['SKIP']}")
    print(f"  MISSING_TAG: {counts['MISSING_TAG']}")
    print(f"  MALFORMED_INCL: {counts['MALFORMED_INCL']}")

    # Detalji (seq_no umesto broja linije)
    print("\n--- DETAILS ---")
    for (seq, name, tag, model, classname, raw) in details:
        mc = f"{model}.{classname}" if (model and classname) else ""
        print(f"{seq:5d} | {name} | {tag} | {mc} | {raw}")

if __name__ == "__main__":
    main()
