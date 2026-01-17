#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import difflib

# ⇩⇩⇩ PODESI OVE PUTANJE ⇩⇩⇩
PATH_SOT = "/home/aleksanovevski/Documents/Projects/intellicupLabelingTool/Blaznavac Venue - Labeler Visual Analysis Template.txt"
PATH_POS = "/home/aleksanovevski/Documents/Projects/intellicupLabelingTool/Blaznavac_cene_artikli_9.25.txt"

# Uključi/isključi fuzzy predloge (True/False)
SUGGEST_FUZZY = True
FUZZY_THRESHOLD = 0.80  # 0..1

# ✅ NOVO: hvataj ime do " - [" ili "[" AKO postoje, INAČE do kraja reda
NAME_RE = re.compile(
    r'art:\s*(?P<name>.+?)(?:\s*(?=\-\s*\[|\[)|\s*$)',
    re.IGNORECASE
)

def norm(s: str) -> str:
    """Normalizacija radi poređenja: spajanje razmaka + casefold + poravnanje crtica."""
    s = (s.replace('\u2010','-').replace('\u2011','-').replace('\u2012','-')
           .replace('\u2013','-').replace('\u2014','-').replace('\u2015','-'))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.casefold()

def parse_sot_names(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    names = []
    for raw in lines:
        line = raw.rstrip()
        if 'art: ' not in line:
            continue
        m = NAME_RE.search(line)
        if m:
            name = m.group("name").strip()
            # ako je neko ostavio " - " bez zagrade, skini završni '-' ako zjapi sam
            name = re.sub(r'\s*-\s*$', '', name)
            if name:
                names.append(name)

    return names

def parse_pos_names(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    names = [re.sub(r'\s+', ' ', ln).strip() for ln in lines]
    names = [n for n in names if n]

    return names

def main():
    sot_names = parse_sot_names(PATH_SOT)
    pos_names = parse_pos_names(PATH_POS)

    # mape za originalna imena po normalizovanom ključu
    sot_map = {}
    for n in sot_names:
        k = norm(n)
        sot_map.setdefault(k, n)
    pos_map = {}
    for n in pos_names:
        k = norm(n)
        pos_map.setdefault(k, n)

    sot_set = set(sot_map.keys())
    pos_set = set(pos_map.keys())

    only_in_pos = sorted(pos_set - sot_set)
    only_in_sot = sorted(sot_set - pos_set)
    in_both = sot_set & pos_set

    print(f"SOT total: {len(sot_set)} unique names")
    print(f"POS total: {len(pos_set)} unique names")
    print(f"Intersection: {len(in_both)}")
    print(f"Only in POS: {len(only_in_pos)}")
    print(f"Only in SOT: {len(only_in_sot)}")

    if only_in_pos:
        print("\n--- ONLY IN POS ---")
        for i, k in enumerate(only_in_pos, 1):
            orig = pos_map[k]
            line = f"{i:4d}. {orig}"
            if SUGGEST_FUZZY and sot_set:
                candidates = difflib.get_close_matches(k, sot_set, n=1, cutoff=FUZZY_THRESHOLD)
                if candidates:
                    best = candidates[0]
                    score = difflib.SequenceMatcher(None, k, best).ratio()
                    line += f"   (closest in SOT: '{sot_map[best]}' ~ {score:.2f})"
            print(line)

    if only_in_sot:
        print("\n--- ONLY IN SOT ---")
        for i, k in enumerate(only_in_sot, 1):
            print(f"{i:4d}. {sot_map[k]}")

if __name__ == "__main__":
    main()
