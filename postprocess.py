import json
import os
import pickle
import sys

import numpy as np

from core import (FIELDS, fuse_doc, norm, parse_date, validity, viz_text)
from features import Vocab, build_row

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT_FIELDS = ("Birth_place_ru", "Issue_organization_ru")


def build_answer(d, fz):
    """Значения 10 полей, если документ прошёл проверки на основе MRZ и срока действия, иначе None."""
    if not all(fz[f]["mrz"] is not None for f in
               ("Licence_number", "Issue_organisation_code", "Birth_date", "Issue_date")):
        return None
    if fz["names"]["state"] not in ("verified", "repaired") or fz["Sex_ru"]["state"] != "verified":
        return None
    b, i = parse_date(fz["Birth_date"]["value"]), parse_date(fz["Issue_date"]["value"])
    if validity(b, i)[0] != "ok":
        return None
    ans = {f: viz_text(d, f) or "" for f in TEXT_FIELDS}
    if not ans["Birth_place_ru"] or not ans["Issue_organization_ru"]:
        return None
    ans.update(fz["names"]["values"])
    for f in ("Sex_ru", "Birth_date", "Issue_date", "Issue_organisation_code", "Licence_number"):
        ans[f] = fz[f]["value"]
    return {f: ans[f] for f in FIELDS}


def main(src, dst):
    with open(os.path.join(HERE, "model", "model.pkl"), "rb") as fh:
        art = pickle.load(fh)
    models, thr = art["models"], art["threshold"]
    with open(os.path.join(HERE, "model", "vocab.json"), encoding="utf-8") as fh:
        vocab = Vocab.from_json(json.load(fh), os.path.join(HERE, "reference", "subdivisions.csv"))
    n_acc = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            decision, fields = "reject", None
            try:
                fz = fuse_doc(d)
                ans = build_answer(d, fz)
                if ans is not None:
                    p = min(np.mean([m.predict_proba(np.array([build_row(d, fz, vocab, f)], float))[0, 1]
                                     for m in models[f]]) for f in TEXT_FIELDS)
                    if p >= thr:
                        decision, fields = "accept", ans
            except Exception:           # любой сбой разбора -> отклоняем
                decision, fields = "reject", None
            rec = {"doc_id": d["doc_id"], "decision": decision}
            if fields is not None:
                rec["fields"] = fields
                n_acc += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"принято {n_acc}, отклонено {sum(1 for _ in open(src, encoding='utf-8')) - n_acc}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("python postprocess.py data/test_predictions.jsonl answers.jsonl")
    main(sys.argv[1], sys.argv[2])
