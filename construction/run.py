import os
import sys
import json
import time
from verifier import TpRef


# Paths will be passed as arguments
DEFAULT_PORT = os.getenv("REFINER_PORT")
DEFAULT_MAX_WORKERS = os.getenv("REFINER_MAX_WORKERS")
DEFAULT_MODEL = os.getenv("REFINER_MODEL")
DEFAULT_MAX_TOKENS = os.getenv("REFINER_MAX_TOKENS")
DEFAULT_HOST = os.getenv("REFINER_HOST", "localhost")

def dedup_row(row, canon):
    # Deduplicate triples within a row (case-insensitive) and reuse canonical casing across rows
    if not row or not isinstance(row, list):
        return []
    seen = set()
    out = []
    for tp in row:
        if not isinstance(tp, (list, tuple)) or len(tp) != 3:
            continue
        s, p, o = tp
        s = s if isinstance(s, str) else (str(s) if s is not None else "")
        p = p if isinstance(p, str) else (str(p) if p is not None else "")
        o = o if isinstance(o, str) else (str(o) if o is not None else "")
        key = (s.lower(), p.lower(), o.lower())
        if key in seen:
            continue
        seen.add(key)
        if key not in canon:
            canon[key] = [s, p, o]
        out.append(canon[key])
    return out

def main():
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python run.py <model_name> <dataset_name> <input_dir> <output_dir>")
        sys.exit(1)
    
    mdl_nm = sys.argv[1]
    dset_nm = sys.argv[2]
    input_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    mw = int(DEFAULT_MAX_WORKERS) if DEFAULT_MAX_WORKERS else 10
    mt = int(DEFAULT_MAX_TOKENS) if DEFAULT_MAX_TOKENS else 10000

    mdl_l = str(mdl_nm).lower()
    mod_l = str(DEFAULT_MODEL or "").lower()
    gpt = mdl_l.startswith("gpt") or mod_l.startswith("gpt")
    if gpt:
        if mdl_l.startswith("gpt"):
            model = "gpt-5.1" if mdl_l == "gpt" else mdl_nm
        else:
            model = DEFAULT_MODEL
            if str(model).lower() == "gpt":
                model = "gpt-5.1"
        ref = TpRef(max_workers=mw, model=model, max_tokens=mt)
    else:
        if not DEFAULT_PORT:
            print("Error: REFINER_PORT is not set.", flush=True)
            raise SystemExit(1)
        ref = TpRef(host=DEFAULT_HOST, port=int(DEFAULT_PORT), max_workers=mw,
                    model=DEFAULT_MODEL, max_tokens=mt)
    
    # Read articles from input folder or file
    if os.path.isfile(input_dir):
        src_p = input_dir
    else:
        src_p = os.path.join(input_dir, "articles.txt")
    
    # Read extract_triples.txt
    prd_p = os.path.join(output_dir, dset_nm, "extract_triples.txt")
    
    if not os.path.exists(src_p):
        print(f"Skip: {src_p} not found")
        return
    if not os.path.exists(prd_p):
        print(f"Skip: {prd_p} not found")
        return
    
    with open(src_p, 'r', encoding='utf-8') as f:
        txts = [l.strip() for l in f.readlines()]
    
    with open(prd_p, 'r', encoding='utf-8') as f:
        preds = [l.strip() for l in f.readlines()]
    
    lim = int(os.getenv("LIM", "0") or "0")
    if lim > 0:
        lim = min(lim, len(txts), len(preds))
        txts = txts[:lim]
        preds = preds[:lim]
    
    # Output directory
    out_dir = os.path.join(output_dir, dset_nm)
    os.makedirs(out_dir, exist_ok=True)
    
    # Output file for refined triples
    refined_p = os.path.join(out_dir, "refined_triples.txt")
    
    # Process and save refined triples
    t0 = time.time()
    outs = ref.proc_batch(txts, preds)
    vtsc = time.time() - t0
    canon = {}

    with open(refined_p, 'w', encoding='utf-8') as f:
        for tps in outs:
            f.write(str(dedup_row(tps, canon)) + '\n')

    # Read extractor stats
    exst = {}
    esf = os.path.join(out_dir, "extract_stats.json")
    if os.path.exists(esf):
        with open(esf, encoding="utf-8") as f:
            exst = json.load(f)
        os.remove(esf)

    # Save structured stats
    tchr = exst.get("tchr", sum(len(t) for t in txts))
    ncll = exst.get("ncll", 0) + ref.ncll
    titk = exst.get("titk", 0) + ref.titk
    totk = exst.get("totk", 0) + ref.totk
    tsec = exst.get("tsec", 0.0) + vtsc

    stats = {
        "extract": {"ncll": exst.get("ncll", 0), "titk": exst.get("titk", 0), "totk": exst.get("totk", 0), "tsec": exst.get("tsec", 0.0)},
        "verify":  {"ncll": ref.ncll, "titk": ref.titk, "totk": ref.totk, "tsec": vtsc},
        "total":   {"ncll": ncll, "titk": titk, "totk": totk, "tchr": tchr, "tsec": tsec},
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print("[Stats Summary]")
    print(f"  Total LLM Calls        : {ncll}")
    print(f"  Total Time (s)         : {tsec:.2f}")
    if tchr > 0:
        print(f"  Input  Tokens / 1k chars : {titk / tchr * 1000:.2f}")
        print(f"  Output Tokens / 1k chars : {totk / tchr * 1000:.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
