import json
from pathlib import Path


def run_eval(data_path, exp_path):
    data = {f["id"]: f for f in json.loads(Path(data_path).read_text())["functions"]}
    exp = json.loads(Path(exp_path).read_text())
    ok = 0
    print(f"{'function':32} {'expected':10} {'got':10} {'conf':7}  note")
    for e in exp["cases"]:
        f = data.get(e["id"])
        got = f["risk"] if f else "missing"
        hit = got == e["expected_risk"]
        ok += hit
        print(f"{e['id'][:32]:32} {e['expected_risk']:10} {got:10} {(f or {}).get('confidence', '-'):7}  {'OK ' if hit else 'MISS'} {e['why'][:70]}")
    print(f"\n{ok}/{len(exp['cases'])} match the human label")
    return 0 if ok == len(exp["cases"]) else 1
