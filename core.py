import datetime as dt
import re

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
TODAY = dt.date(2026, 10, 1)
REPAIR_MARGIN = 0.0
REPAIR_MIN_GAP = 0.05   # минимальная разница уверенностей viz и MRZ для ремонта символа

# Транслитерация МВД для MRZ внутреннего паспорта РФ (проверено на train)
TR = dict(zip("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", "ABVGDE2JZIQKLMNOPRSTUFHC34WXY9678"))
INV = {v: k for k, v in TR.items()}


def norm(s):
    return " ".join(str(s or "").split()).upper()


# ---------- визуальная зона ----------
def viz_text(d, f):
    v = d["viz"].get(f)
    if not v or not v.get("lines"):
        return None
    return norm(" ".join(l["text"] for l in v["lines"]))


def viz_chars(d, f):
    """char_conf всех строк поля одним списком."""
    v = d["viz"].get(f)
    if not v or not v.get("lines"):
        return []
    out = []
    for l in v["lines"]:
        out.extend(l["char_conf"])
    return out


def parse_date(s):
    if not s or not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", s):
        return None
    try:
        return dt.date(int(s[6:]), int(s[3:5]), int(s[:2]))
    except ValueError:
        return None


def fmt_date(x):
    return x.strftime("%d.%m.%Y")


def viz_number(s):
    if not s:
        return None
    digits = re.sub(r"\s", "", s)
    if re.fullmatch(r"\d{10}", digits):
        return f"{digits[:2]} {digits[2:4]} {digits[4:]}"
    return None


def viz_code(s):
    return s if s and re.fullmatch(r"\d{3}-\d{3}", s) else None


# ---------- MRZ ----------
def check_digit(s):
    w = (7, 3, 1)
    t = 0
    for i, c in enumerate(s):
        if c == "<":
            v = 0
        elif c.isdigit():
            v = int(c)
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        else:
            return None
        t += v * w[i % 3]
    return t % 10


def tr(s):
    """Кириллица -> MRZ; пробел и дефис -> '<'. None, если есть неизвестный символ."""
    out = []
    for c in s:
        if c in " -":
            out.append("<")
        elif c in TR:
            out.append(TR[c])
        else:
            return None
    return "".join(out)


def yymmdd(s):
    if not re.fullmatch(r"\d{6}", s or ""):
        return None
    return int(s[:2]), int(s[2:4]), int(s[4:])


def parse_mrz(d):
    """Возвращает словарь с l1/l2 и результатами контрольных цифр."""
    r = {"l1": None, "l2": None, "c1": [], "c2": [], "n_lines": 0,
         "num_ok": False, "birth_ok": False, "opt_ok": False, "comp_ok": False, "sex": None,
         "number": None, "birth": None, "issue": None, "code": None}
    m = d.get("mrz")
    if not m or not m.get("lines"):
        return r
    r["n_lines"] = len(m["lines"])
    for ln in m["lines"]:
        t = ln["text"]
        if len(t) == 44 and t[10:13] == "RUS" and re.match(r"[0-9<]", t[:1] or "x"):
            r["l2"], r["c2"] = t, ln["char_conf"]
        elif t.startswith("P") and "<<" in t:
            r["l1"], r["c1"] = t, ln["char_conf"]
    l2 = r["l2"]
    if l2:
        r["num_ok"] = check_digit(l2[0:9]) is not None and str(check_digit(l2[0:9])) == l2[9]
        r["birth_ok"] = check_digit(l2[13:19]) is not None and str(check_digit(l2[13:19])) == l2[19]
        r["opt_ok"] = (check_digit(l2[28:42]) is not None and str(check_digit(l2[28:42])) == l2[42]
                       and l2[41] == "<")
        comp = l2[0:10] + l2[13:20] + l2[21:43]
        r["comp_ok"] = check_digit(comp) is not None and str(check_digit(comp)) == l2[43]
        r["sex"] = {"M": "МУЖ.", "F": "ЖЕН."}.get(l2[20])
        if r["num_ok"] and r["opt_ok"] and l2[28].isdigit() and l2[0:9].isdigit():
            r["number"] = f"{l2[0:2]} {l2[2]}{l2[28]} {l2[3:9]}"
        if r["birth_ok"]:
            r["birth"] = yymmdd(l2[13:19])
        if r["opt_ok"]:
            iss = yymmdd(l2[29:35])
            if iss and re.fullmatch(r"\d{6}", l2[35:41]):
                r["issue"] = iss
                r["code"] = f"{l2[35:38]}-{l2[38:41]}"
    return r


def birth_from_mrz(yy, mm, dd, viz_date=None):
    """Год рождения по двум цифрам: окно 1927..2012 (возраст паспорта 14..100 лет)."""
    cands = []
    for cent in (1900, 2000):
        try:
            cands.append(dt.date(cent + yy, mm, dd))
        except ValueError:
            pass
    ok = [c for c in cands if dt.date(1927, 1, 1) <= c <= dt.date(2012, 12, 31)]
    if viz_date is not None and viz_date in cands:
        return viz_date
    return ok[0] if len(ok) == 1 else None


def issue_from_mrz(yy, mm, dd):
    cands = []
    for cent in (1900, 2000):
        try:
            c = dt.date(cent + yy, mm, dd)
        except ValueError:
            continue
        if dt.date(1996, 1, 1) <= c <= TODAY:
            cands.append(c)
    return cands[0] if len(cands) == 1 else None


def mrz_expected_line1(last, first, middle):
    a, b = tr(last or ""), tr(first or "")
    if a is None or b is None or not last or not first:
        return None
    body = a + "<<" + b
    if middle:
        c = tr(middle)
        if c is None:
            return None
        body += "<" + c
    return body


# ---------- действительность ----------
def add_years(d, n):
    try:
        return d.replace(year=d.year + n)
    except ValueError:
        return d.replace(year=d.year + n, day=28)


def validity(birth, issue):
    """('ok'|'expired'|'grace'|'bad', дата окончания или None).
    Паспорт меняют в 20 и 45 лет. Консервативно: действителен, только если дата
    замены строго после TODAY; льготные 90 дней после дня рождения не считаем действительными."""
    if birth is None or issue is None:
        return "bad", None
    if issue > TODAY or issue < add_years(birth, 14):
        return "bad", None
    if issue < add_years(birth, 20):
        lim = add_years(birth, 20)
    elif issue < add_years(birth, 45):
        lim = add_years(birth, 45)
    else:
        return "ok", None
    if lim > TODAY:
        return "ok", lim
    return ("grace" if (TODAY - lim).days <= 90 else "expired"), lim


# ---------- слияние визуальной зоны и MRZ ----------
def fuse_doc(d):
    """По 8 полям, у которых есть вторая независимая оценка (MRZ).
    Для каждого поля: value, state in {'verified','mrz_only','viz_only','none','conflict'}."""
    m = parse_mrz(d)
    out = {}
    V = {f: viz_text(d, f) for f in FIELDS}

    # номер
    vn = viz_number(V["Licence_number"])
    mn = m["number"]
    out["Licence_number"] = _pick(vn, mn)

    # код подразделения
    vc = viz_code(V["Issue_organisation_code"])
    mc = m["code"]
    out["Issue_organisation_code"] = _pick(vc, mc)

    # даты
    vb = parse_date(V["Birth_date"])
    mb = None
    if m["birth"]:
        yy, mm, dd = m["birth"]
        mb = birth_from_mrz(yy, mm, dd, vb)
    out["Birth_date"] = _pick(fmt_date(vb) if vb else None, fmt_date(mb) if mb else None)
    vi = parse_date(V["Issue_date"])
    mi = None
    if m["issue"]:
        yy, mm, dd = m["issue"]
        mi = issue_from_mrz(yy, mm, dd)
    out["Issue_date"] = _pick(fmt_date(vi) if vi else None, fmt_date(mi) if mi else None)

    # ФИО: сверяем всю строку MRZ-1 с транслитом визуальных значений
    last, first, mid = V["Last_name_ru"], V["First_name_ru"], V["Middle_name_ru"]
    exp = mrz_expected_line1(last, first, mid or "")
    l1 = m["l1"]
    names_state = "none"
    if exp is not None and l1:
        got = l1[5:].rstrip("<")
        want = exp.rstrip("<")
        if len(want) + 5 > 44:       # строка усечена: хвост не проверить
            names_state = "truncated"
        elif got == want:
            names_state = "verified"
        else:
            names_state = "conflict"
    elif exp is not None:
        names_state = "viz_only"
    elif l1:
        names_state = "mrz_only"
    values = {"Last_name_ru": last or "", "First_name_ru": first or "", "Middle_name_ru": mid or ""}
    if names_state == "conflict":
        rep, info = name_repair(d, REPAIR_MARGIN)
        if rep is not None and isinstance(info, list) and info and \
                min(abs(pv - pm) for _, _, pv, pm, _ in info) >= REPAIR_MIN_GAP:
            names_state, values = "repaired", rep
    out["names"] = {"state": names_state, "mid_empty": not mid, "values": values}

    # пол: viz, MRZ и окончание отчества
    ps = None
    if mid:
        if re.search(r"(ВИЧ|ОГЛЫ|УЛЫ)$", mid):
            ps = "МУЖ."
        elif re.search(r"(ВНА|ЧНА|КЫЗЫ|КИЗИ)$", mid):
            ps = "ЖЕН."
    vs = V["Sex_ru"] if V["Sex_ru"] in ("МУЖ.", "ЖЕН.") else None
    out["Sex_ru"] = _pick3(vs, m["sex"], ps)
    out["_mrz"] = m
    return out


def _pick(v, m):
    if v is not None and m is not None:
        # при расхождении берём MRZ: значение прошло контрольные цифры (на train 100% верно)
        return {"value": m, "state": "verified" if v == m else "conflict", "viz": v, "mrz": m}
    if m is not None:
        return {"value": m, "state": "mrz_only", "viz": v, "mrz": m}
    if v is not None:
        return {"value": v, "state": "viz_only", "viz": v, "mrz": m}
    return {"value": None, "state": "none", "viz": v, "mrz": m}


def _pick3(a, b, c):
    vals = [x for x in (a, b, c) if x is not None]
    if not vals:
        return {"value": None, "state": "none", "agree": 0}
    top = max(set(vals), key=vals.count)
    agree = vals.count(top)
    return {"value": top, "state": "verified" if agree >= 2 and agree == len(vals) else
            ("majority" if agree >= 2 else "single"), "agree": agree, "n": len(vals)}


# ---------- ремонт ФИО по символьному согласованию viz и MRZ ----------
def _name_layout(last, first, mid):
    """Список (поле, индекс символа) для каждой позиции ожидаемой строки MRZ-1 (после 'PNRUS')."""
    lay = [("L", i) for i in range(len(last))] + [("sep", 0)] * 2 + [("F", i) for i in range(len(first))]
    if mid:
        lay += [("sep", 0)] + [("M", i) for i in range(len(mid))]
    return lay


def name_repair(d, margin=0.0):
    """Для расхождения viz/MRZ одинаковой длины выбираем по символам тот, у кого выше уверенность.
    Возвращает (names dict | None, info)."""
    m = parse_mrz(d)
    V = {k: viz_text(d, k) for k in ("Last_name_ru", "First_name_ru", "Middle_name_ru")}
    last, first, mid = V["Last_name_ru"], V["First_name_ru"], V["Middle_name_ru"] or ""
    if not last or not first or not m["l1"]:
        return None, "nodata"
    exp = mrz_expected_line1(last, first, mid)
    if exp is None:
        return None, "badchar"
    B = m["l1"][5:].rstrip("<")
    E = exp.rstrip("<")
    if len(B) != len(E) or len(E) + 5 > 44:
        return None, "length"
    lay = _name_layout(last, first, mid)
    # доверие viz-символов (индексы по тексту поля: line breaks не учитываем -> только однострочные имена)
    conf = {}
    for key, k in (("L", "Last_name_ru"), ("F", "First_name_ru"), ("M", "Middle_name_ru")):
        cc = viz_chars(d, k)
        if cc and len(cc) == len(V[k] or ""):
            conf[key] = cc
        else:
            conf[key] = None
    c1 = m["c1"]
    out = {"L": list(last), "F": list(first), "M": list(mid)}
    diffs = []
    for i, (a, b) in enumerate(zip(E, B)):
        if a == b:
            continue
        fld, j = lay[i] if i < len(lay) else ("sep", 0)
        if fld == "sep" or b not in INV or conf[fld] is None or 5 + i >= len(c1):
            return None, "ambiguous"
        pv, pm = conf[fld][j], c1[5 + i]
        diffs.append((fld, j, pv, pm, INV[b]))
    for fld, j, pv, pm, mch in diffs:
        if pm - pv > margin:
            out[fld][j] = mch
        elif pv - pm <= margin:
            return None, "tie"
    res = {"Last_name_ru": "".join(out["L"]), "First_name_ru": "".join(out["F"]),
           "Middle_name_ru": "".join(out["M"])}
    return res, diffs
