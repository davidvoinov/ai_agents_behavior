#!/usr/bin/env python3
"""Remove duplicate dyad_id records left by two runners writing the same file.
Keeps the first occurrence. Back up first. Run only when no runner is active."""
import json, sys, shutil
for path in sys.argv[1:]:
    rows=[json.loads(l) for l in open(path) if l.strip()]
    seen=set(); keep=[]
    for r in rows:
        if r["dyad_id"] in seen: continue
        seen.add(r["dyad_id"]); keep.append(r)
    if len(keep)==len(rows):
        print(f"{path}: no duplicates"); continue
    shutil.copy(path, path+".predupe")
    with open(path,"w") as f:
        for r in keep: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"{path}: {len(rows)} -> {len(keep)} (removed {len(rows)-len(keep)})")
