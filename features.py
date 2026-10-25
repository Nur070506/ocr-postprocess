import collections
import csv
import re

import numpy as np

from core import FIELDS, norm, viz_chars, viz_text, fuse_doc

TOK = re.compile(r"[А-ЯЁ0-9]+(?:-[А-ЯЁ0-9]+)*")


def tokens(s):
    return TOK.findall(s or "")


def lev(a, b, cap=99):
    """Расстояние Левенштейна (короткие строки)."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_CSV_CACHE = {}


def _load_csv(path):
    if path not in _CSV_CACHE:
        code = collections.defaultdict(set)
        tok = collections.Counter()
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                n = norm(r["name"])
                code[r["code"]].add(n)
                for t in tokens(n):
                    tok[t] += 1
        _CSV_CACHE[path] = (code, tok)
    return _CSV_CACHE[path]


class Vocab:
    """Словари, собранные по разметке train (и справочнику подразделений)."""

    def __init__(self, labels, csv_path):
        self.place_tok = collections.Counter()
        self.org_tok = collections.Counter()
        self.place_full = collections.Counter()
        self.org_full = collections.Counter()
        self.code_orgs = collections.defaultdict(collections.Counter)
        self.place_last = collections.Counter()
        self.org_last = collections.Counter()
        self.raw = {"Birth_place_ru": collections.Counter(), "Issue_organization_ru": collections.Counter()}
        self.rawfirst = {"Birth_place_ru": collections.Counter(), "Issue_organization_ru": collections.Counter()}
        self.rawlast = {"Birth_place_ru": collections.Counter(), "Issue_organization_ru": collections.Counter()}
        for g in labels:
            self.add(g, +1)
        self.csv_code, self.csv_tok = _load_csv(csv_path)


    # --- сериализация (JSON), чтобы не зависеть от pickle для словарей ---
    _KEYS = ("place_tok", "org_tok", "place_full", "org_full", "place_last", "org_last")

    def to_json(self):
        d = {k: dict(getattr(self, k)) for k in self._KEYS}
        d["code_orgs"] = {c: dict(v) for c, v in self.code_orgs.items()}
        for k in ("raw", "rawfirst", "rawlast"):
            d[k] = {f: dict(c) for f, c in getattr(self, k).items()}
        return d

    @classmethod
    def from_json(cls, d, csv_path):
        self = cls.__new__(cls)
        for k in cls._KEYS:
            setattr(self, k, collections.Counter(d[k]))
        self.code_orgs = collections.defaultdict(collections.Counter,
                                                 {c: collections.Counter(v) for c, v in d["code_orgs"].items()})
        for k in ("raw", "rawfirst", "rawlast"):
            setattr(self, k, {f: collections.Counter(c) for f, c in d[k].items()})
        self.csv_code, self.csv_tok = _load_csv(csv_path)
        return self

    def add(self, g, sign):
        for f in ("Birth_place_ru", "Issue_organization_ru"):
            rt = norm(g[f]).split(" ")
            for t in rt:
                self.raw[f][t] += sign
            self.rawfirst[f][rt[0]] += sign
            self.rawlast[f][rt[-1]] += sign
        self.place_full[norm(g["Birth_place_ru"])] += sign
        self.org_full[norm(g["Issue_organization_ru"])] += sign
        self.code_orgs[g["Issue_organisation_code"]][norm(g["Issue_organization_ru"])] += sign
        tp, to = tokens(norm(g["Birth_place_ru"])), tokens(norm(g["Issue_organization_ru"]))
        if tp:
            self.place_last[tp[-1]] += sign
        if to:
            self.org_last[to[-1]] += sign
        for t in tp:
            self.place_tok[t] += sign
        for t in tokens(norm(g["Issue_organization_ru"])):
            self.org_tok[t] += sign


def token_confs(d, f):
    """[(token, min char_conf)] по строкам поля."""
    v = d["viz"].get(f)
    out = []
    if not v:
        return out
    for l in v["lines"]:
        t = norm(l["text"])
        cc = l["char_conf"]
        # norm мог сжать пробелы -> индексы считаем по исходному тексту
        raw = l["text"].upper()
        for m in TOK.finditer(raw):
            seg = cc[m.start():m.end()]
            out.append((m.group(), min(seg) if seg else 0.0))
    return out


def generic(d, f):
    cc = np.array(viz_chars(d, f) or [0.0])
    v = d["viz"].get(f) or {}
    lines = v.get("lines", [])
    gap = hl = 0.0
    if lines and v.get("box"):
        lh = float(np.median([l["box"][3] for l in lines])) or 1.0
        bot = max(l["box"][1] + l["box"][3] for l in lines)
        top = min(l["box"][1] for l in lines)
        gap = (v["box"][1] + v["box"][3] - bot) / lh
        hl = v["box"][3] / lh
        topgap = (top - v["box"][1]) / lh
    else:
        topgap = 0.0
    return [
        gap, hl, topgap,
        len(v.get("lines", [])), len(cc), float(v.get("conf", 0.0)),
        cc.min(), cc.mean(), np.percentile(cc, 5),
        (cc < 0.999).sum(), (cc < 0.99).sum(), (cc < 0.95).sum(), (cc < 0.9).sum(),
        np.sort(cc)[:2].sum() if len(cc) >= 2 else cc.sum(),
    ]


def lex_features(d, f, counter, code_cands=None):
    toks = token_confs(d, f)
    n = len(toks)
    oov = [(t, c) for t, c in toks if counter.get(t, 0) < 1]
    rare = [(t, c) for t, c in toks if counter.get(t, 0) < 3]
    feats = [n, len(oov), len(rare),
             min([c for _, c in oov], default=1.0), min([c for _, c in rare], default=1.0),
             sum(1 for t, _ in toks if t in (d.get("_csvtok") or ()))]
    return feats


def doc_features(d, fz):
    """Признаки документа в целом: качество снимка, согласованность остальных полей."""
    q = d["quality"]
    m = fz["_mrz"]
    states = [fz[f]["state"] for f in ("Licence_number", "Issue_organisation_code", "Birth_date", "Issue_date")]
    other_min = min([min(viz_chars(d, f) or [0.0]) for f in FIELDS if f not in
                     ("Birth_place_ru", "Issue_organization_ru")] or [0.0])
    return [
        q["sharpness"], q["doc_px"], q["glare"],
        {"web": 0, "mobile": 1, "mobile_v2": 2}[d["source"]],
        sum(s == "conflict" for s in states), sum(s in ("viz_only", "none") for s in states),
        fz["names"]["state"] == "conflict", fz["names"]["state"] == "repaired", m["n_lines"], m["num_ok"], m["birth_ok"], m["opt_ok"], m["comp_ok"],
        other_min,
    ]


def text_feats(d, f, fz, vocab):
    """Лексические признаки поля места рождения / органа выдачи."""
    t = viz_text(d, f) or ""
    toks = token_confs(d, f)
    counter = vocab.place_tok if f == "Birth_place_ru" else vocab.org_tok
    oov = [(x, c) for x, c in toks if counter.get(x, 0) < 1 and vocab.csv_tok.get(x, 0) < 1]
    rare = [(x, c) for x, c in toks if counter.get(x, 0) + vocab.csv_tok.get(x, 0) < 3]
    csvk = [x for x, _ in toks if vocab.csv_tok.get(x, 0) > 0]
    last = toks[-1][0] if toks else ""
    lastc = (vocab.place_last if f == "Birth_place_ru" else vocab.org_last).get(last, 0)
    feats = [lastc, np.log1p(max(lastc, 0)), len(toks), len(oov), len(rare), len(csvk),
             min([c for _, c in oov], default=1.0), min([c for _, c in rare], default=1.0)]
    if f == "Birth_place_ru":
        feats += [vocab.place_full.get(t, 0) > 0]
    else:
        code = fz["Issue_organisation_code"]["value"]
        cands = {k for k, v in vocab.code_orgs.get(code, {}).items() if v > 0} | vocab.csv_code.get(code, set())
        feats += [vocab.org_full.get(t, 0) > 0,
                  t in {k for k, v in vocab.code_orgs.get(code, {}).items() if v > 0},
                  t in vocab.csv_code.get(code, set()),
                  len(cands) > 0,
                  min([lev(t, c, 12) / max(len(t), 1) for c in cands], default=1.0)]
    return feats


def build_row(d, fz, vocab, f):
    return generic(d, f) + text_feats(d, f, fz, vocab) + raw_feats(d, f, vocab) + doc_features(d, fz)


RAWTOK = re.compile(r"\S+")


def raw_token_confs(d, f):
    v = d["viz"].get(f)
    out = []
    if not v:
        return out
    for l in v["lines"]:
        raw = l["text"].upper()
        cc = l["char_conf"]
        for m in RAWTOK.finditer(raw):
            seg = cc[m.start():m.end()]
            out.append((m.group(), min(seg) if seg else 0.0))
    return out


def raw_feats(d, f, vocab):
    toks = raw_token_confs(d, f)
    if not toks:
        return [0, 0, 0, 1.0, 0, 0]
    cnt = vocab.raw[f]
    oov = [(t, c) for t, c in toks if cnt.get(t, 0) < 1]
    rare = [t for t, c in toks if cnt.get(t, 0) < 3]
    return [np.log1p(max(vocab.rawfirst[f].get(toks[0][0], 0), 0)), np.log1p(max(vocab.rawlast[f].get(toks[-1][0], 0), 0)),
            len(oov), min([c for _, c in oov], default=1.0), len(rare), len(toks)]
