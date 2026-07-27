import json
import glob

from benchmaxxing.extract import parse_mcq_choice, Abstention
import re

def _letters(n: int) -> str:
    return "".join(chr(ord("A") + i) for i in range(n))

def _legacy_parse(text, options):
    if not text:
        return ""
    t = text.strip()
    letters = _letters(len(options))
    m = re.findall(r"\\boxed\{\s*([A-Z])\s*\}", t)
    if not m:
        m = re.findall(r"(?:final answer|the answer|answer)\s*(?:is|:)?\s*\**\(?([A-Z])\)?\b", t, re.I)
    if m and m[-1].upper() in letters:
        return options[letters.index(m[-1].upper())]
    low = t.lower()
    hits = [(low.rfind(o.lower()), o) for o in options if o.lower() in low]
    hits = [(p, o) for p, o in hits if p >= 0]
    if hits:
        return max(hits)[1]
    m2 = re.search(r"\b([A-Z])\b\s*[.)]?\s*$", t.upper())
    if m2 and m2.group(1) in letters:
        return options[letters.index(m2.group(1))]
    if len(t) == 1 and t.upper() in letters:
        return options[letters.index(t.upper())]
    return ""

def parse_legacy_string(text, options):
    val = parse_mcq_choice(text, options)
    if isinstance(val, Abstention):
        return ""
    return options[val]

def run():
    files = glob.glob("experiments/*/results/*cache*.jsonl")
    
    total = 0
    mismatches = 0
    
    for path in files:
        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                resp = obj.get("resp", "")
                
                opts = ["A", "B", "C", "D", "E"]
                
                leg = _legacy_parse(resp, opts)
                new_ = parse_legacy_string(resp, opts)
                
                if leg != new_:
                    mismatches += 1
                    if mismatches <= 5:
                        print("\nMISMATCH!")
                        print(f"Legacy  : {leg!r}")
                        print(f"New     : {new_!r}")
                        print(f"Text    : {resp[-100:]}")
                        
                total += 1
                
    print(f"\nTotal: {total}, Mismatches: {mismatches}")

if __name__ == "__main__":
    run()
