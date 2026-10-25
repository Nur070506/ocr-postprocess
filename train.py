import json
import pickle
import random
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from core import FIELDS, fuse_doc, norm, parse_date, validity, viz_text
from features import Vocab, build_row

DATA, REF, OUT = "data", "reference/subdivisions.csv", "model"
K, SEED, THRESHOLD = 5, 0, 0.99
TEXT_FIELDS = ("Birth_place_ru", "Issue_organization_ru")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def make_model():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                          l2_regularization=1.0, random_state=0)


def main():
    P = load(f"{DATA}/train_predictions.jsonl")
    labels = {d["doc_id"]: d["fields"] for d in load(f"{DATA}/train_labels.jsonl")}
    n = len(P)
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    fold = np.zeros(n, int)
    for k, i in enumerate(idx):
        fold[i] = k % K
    FZ = [fuse_doc(d) for d in P]
    X = {f: [None] * n for f in TEXT_FIELDS}
    for k in range(K):
        V = Vocab([labels[P[i]["doc_id"]] for i in range(n) if fold[i] != k], REF)
        for i in range(n):
            if fold[i] == k:
                for f in TEXT_FIELDS:
                    X[f][i] = build_row(P[i], FZ[i], V, f)
    X = {f: np.array(v, float) for f, v in X.items()}
    Y = {f: np.array([viz_text(P[i], f) == norm(labels[P[i]["doc_id"]][f]) for i in range(n)]) for f in TEXT_FIELDS}

    models = {f: [] for f in TEXT_FIELDS}
    oof = {f: np.zeros(n) for f in TEXT_FIELDS}
    for f in TEXT_FIELDS:
        for k in range(K):
            m = make_model().fit(X[f][fold != k], Y[f][fold != k])
            models[f].append(m)
            oof[f][fold == k] = m.predict_proba(X[f][fold == k])[:, 1]

    # out-of-fold оценка решения по документу
    from postprocess import build_answer   # тот же код, что и на инференсе
    score = np.minimum(oof[TEXT_FIELDS[0]], oof[TEXT_FIELDS[1]])
    print("порог  принято  ошибочных  (out-of-fold)")
    for thr in (0.999, 0.998, 0.995, 0.99, 0.98, 0.95):
        acc = wrong = 0
        for i in range(n):
            if score[i] < thr:
                continue
            ans = build_answer(P[i], FZ[i])
            if ans is None:
                continue
            acc += 1
            wrong += any(norm(ans[f]) != norm(labels[P[i]["doc_id"]][f]) for f in FIELDS)
        print(f"{thr:<6} {acc:>7} {wrong:>9}")

    import os
    os.makedirs(OUT, exist_ok=True)
    full = Vocab(list(labels.values()), REF)
    with open(f"{OUT}/vocab.json", "w", encoding="utf-8") as fh:
        json.dump(full.to_json(), fh, ensure_ascii=False)
    with open(f"{OUT}/model.pkl", "wb") as fh:
        pickle.dump({"models": models, "threshold": THRESHOLD}, fh)
    print("сохранено в", OUT)


if __name__ == "__main__":
    sys.exit(main())
