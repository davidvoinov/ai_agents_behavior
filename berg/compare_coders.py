import json, sys, collections
import code_comm as CC

def load(p):
    return {json.loads(l)["uid"]: json.loads(l) for l in open(p) if l.strip()}

def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = load(a_path), load(b_path)
    shared = sorted(set(A) & set(B))
    print(f"coder A: {len(A)}  coder B: {len(B)}  overlap: {len(shared)}")
    if not shared:
        print("no overlapping uids — run the judge on the sampled transcripts first"); return
    a=[A[u] for u in shared]; b=[B[u] for u in shared]
    print(f"\n{'variable':22}{'kappa':>8}{'raw agree':>11}")
    for v in CC.CATEGORICAL:
        k = CC.cohen_kappa([x.get(v) for x in a], [x.get(v) for x in b])
        raw = sum(1 for x,y in zip(a,b) if x.get(v)==y.get(v))/len(shared)
        print(f"{v:22}{k:>8.3f}{raw:>11.1%}")
    for v in ["agreed_send","agreed_return"]:
        both=[(x.get(v),y.get(v)) for x,y in zip(a,b) if x.get(v) is not None and y.get(v) is not None]
        if both:
            ex=sum(1 for p,q in both if p==q)/len(both)
            print(f"{v:22}{'--':>8}{ex:>11.1%}  (n={len(both)} both non-null)")
    print("\ndisagreements on explicit_agreement:")
    n=0
    for u in shared:
        if A[u].get("explicit_agreement")!=B[u].get("explicit_agreement"):
            print(f"  {u}  A={A[u].get('explicit_agreement')}  B={B[u].get('explicit_agreement')}"); n+=1
    print(f"  total {n}")

if __name__ == "__main__":
    main()
