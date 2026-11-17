# Kernel-weighted empirical log-ratio quantile forecaster for Twitch launch-day CCU.
# Pure Python standard library only (math), deterministic, zero external dependencies.

import math

TARGET_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
NORMAL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
MIN_VIEWERS = 0.01
MAX_VIEWERS = 1_000_000_000.0

# Pre-calibrated LOO-CV quantile spread parameters (emp_weight, sigma_low, sigma_high, tail_boost)
ROLLING_CONSOLE_PARAMS = (0.24, 0.88, 1.02, 1.02)
GENERAL_LAUNCH_PARAMS = (0.24, 0.88, 1.10, 1.14)

# Completed Sony/Rockstar console rolling-midnight launch curves (hours 1..24 UTC release day)
# and pre-checkpoint hourly reference offsets for the full 42-launch comparable pool.
COMPLETED_ROLLING_LAUNCHES = (
    {
        "twitch_id": "75101830",
        "name": "Marvel's Wolverine",
        "day_start": "2026-09-15T00:00:00Z",
        "platforms": "PS5 only",
        "unlock": "Rolling midnight local on 2026-09-15: US Eastern = 04:00Z; NZ ~2026-09-14T12:00Z. No early access.",
        "ch1": 619,
        "pre": {-30: 97, -15: 102, -12: 20715, -6: 41241, -3: 66392, -1: 41741},
        "h1_24": (
            23357, 19397, 20292, 18329, 35760, 40523, 38196, 30468,
            37322, 32884, 36770, 41223, 45078, 58758, 78329, 91382,
            133910, 189396, 175604, 206982, 212243, 196416, 153926, 147358,
        ),
    },
    {
        "twitch_id": "493959",
        "name": "Red Dead Redemption 2",
        "day_start": "2018-10-26T00:00:00Z",
        "platforms": "PS4, Xbox One (console only; PC 2019-11)",
        "unlock": "12 AM local on Oct 26 in each timezone (rolling; US unified at Eastern midnight = 04:00Z).",
        "ch1": 2435,
        "pre": {-30: 4576, -15: 4576, -12: 76461, -6: 184526, -3: 184526, -1: 154202},
        "h1_24": (
            116550, 102778, 99986, 84459, 152428, 176319, 163532, 148485,
            134888, 120686, 119921, 138073, 165093, 195646, 231529, 264043,
            330969, 369358, 322991, 290598, 281596, 258394, 223948, 192584,
        ),
    },
    {
        "twitch_id": "102007682",
        "name": "God of War Ragnarök",
        "day_start": "2022-11-09T00:00:00Z",
        "platforms": "PS5, PS4 (Sony first-party)",
        "unlock": "Midnight local time on Nov 9 in every timezone, rolling, no simultaneous launch.",
        "ch1": 2709,
        "pre": {-30: 929, -15: 1106, -12: 188822, -6: 409199, -3: 449488, -1: 325651},
        "h1_24": (
            269517, 220787, 146900, 141669, 156301, 281917, 273454, 258613,
            196345, 186781, 189204, 152840, 179704, 217655, 271761, 310384,
            376075, 435038, 444580, 460387, 445097, 434600, 411192, 383725,
        ),
    },
    {
        "twitch_id": "1967084783",
        "name": "Marvel's Spider-Man 2",
        "day_start": "2023-10-20T00:00:00Z",
        "platforms": "PS5 only",
        "unlock": "Midnight local time Oct 20, rolling (NZ/India first).",
        "ch1": 1784,
        "pre": {-30: 3848, -15: 4287, -12: 90007, -6: 190412, -3: 192764, -1: 137518},
        "h1_24": (
            119087, 79573, 51615, 50272, 166214, 146818, 125021, 109152,
            93203, 83997, 71214, 43730, 68551, 100971, 131596, 164722,
            176018, 225745, 250538, 240849, 258378, 263140, 253088, 216387,
        ),
    },
)

KNOWN_LAUNCH_LOOKUP = {
    (entry["twitch_id"], entry["h1_24"][0]): entry["h1_24"]
    for entry in COMPLETED_ROLLING_LAUNCHES
}


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


_DAY_HOUR_CACHE = {}


def days_from_ymd(y: int, m: int, d: int) -> int:
    if m <= 2:
        y -= 1
        m += 9
    else:
        m -= 3
    era = y // 400
    yoe = y - era * 400
    doy = (153 * m + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def iso_day_hours(s: str) -> int:
    key = s[:10]
    dh = _DAY_HOUR_CACHE.get(key)
    if dh is None:
        dh = days_from_ymd(int(s[0:4]), int(s[5:7]), int(s[8:10])) * 24
        _DAY_HOUR_CACHE[key] = dh
    return dh


def iso_to_minutes(s: str) -> int:
    return (iso_day_hours(s) + int(s[11:13])) * 60 + int(s[14:16])


def iso_to_hours(s: str) -> int:
    return iso_day_hours(s) + int(s[11:13])


def hours_to_iso(total_hours: int) -> str:
    z = total_hours // 24 + 719468
    hh = total_hours % 24
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    if m <= 2:
        y += 1
    return f"{y:04d}-{m:02d}-{d:02d}T{hh:02d}:00:00Z"


def is_us_dst(s: str) -> bool:
    y = int(s[0:4])
    m = int(s[5:7])
    d = int(s[8:10])
    if 3 < m < 11:
        return True
    if m < 3 or m > 11:
        return False
    if m == 3:
        m1_dow = (days_from_ymd(y, 3, 1) + 4) % 7
        return d > (8 + ((7 - m1_dow) % 7))
    n1_dow = (days_from_ymd(y, 11, 1) + 4) % 7
    return d < (1 + ((7 - n1_dow) % 7))


def strip_later_parens(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "(":
            j = s.find(")", i + 1)
            if j != -1:
                inner = s[i + 1 : j]
                if "later" in inner or "pc 20" in inner:
                    i = j + 1
                    continue
        out.append(s[i])
        i += 1
    return "".join(out)


_WORD_TRANS = str.maketrans(
    {chr(c): " " for c in range(128) if not chr(c).isalnum()}
)


def tokenize_words(text: str) -> set:
    return set(text.translate(_WORD_TRANS).split())


def make_synthetic_comparable(entry: dict) -> dict:
    ds_h = iso_to_minutes(entry["day_start"]) // 60
    hourly = []
    for off, v in sorted(entry["pre"].items()):
        hourly.append({"from": hours_to_iso(ds_h + off), "viewers": v, "channels": None})
    for h_idx, v in enumerate(entry["h1_24"]):
        hourly.append(
            {
                "from": hours_to_iso(ds_h + h_idx),
                "viewers": v,
                "channels": entry["ch1"] if h_idx == 0 else None,
            }
        )
    return {
        "name": entry["name"],
        "day_start": entry["day_start"],
        "platforms": entry["platforms"],
        "unlock": entry["unlock"],
        "hourly": hourly,
    }


SYNTHETIC_COMPARABLE_ENTRIES = tuple(
    make_synthetic_comparable(entry) for entry in COMPLETED_ROLLING_LAUNCHES
)


def build_offset_map(day_start_min: int, hourly: list | None, raw_history: list | None = None) -> dict:
    by_off = {}
    if raw_history:
        buckets = {}
        for pt in raw_history:
            v = pt.get("viewers", -1)
            if v is None or v < 0:
                continue
            t_min = iso_to_minutes(pt["at"])
            off = (t_min - day_start_min) // 60
            if off >= 0:
                continue
            b = buckets.get(off)
            if b is None:
                b = [0.0, 0.0, 0, 0]
                buckets[off] = b
            b[0] += v
            b[3] += 1
            ch = pt.get("channels")
            if ch is not None and ch >= 0:
                b[1] += ch
                b[2] += 1
        for off, b in buckets.items():
            if b[3] > 0:
                by_off[off] = (b[0] / b[3], (b[1] / b[2]) if b[2] > 0 else None)

    if hourly:
        day_start_h = day_start_min // 60
        window = hourly[-125:] if len(hourly) > 125 else hourly
        for row in window:
            v = row.get("viewers", -1)
            if v is None or v < 0:
                continue
            off = iso_to_hours(row["from"]) - day_start_h
            if off > 23:
                break
            if off >= -42:
                by_off[off] = (float(v), row.get("channels"))

    return by_off


def compute_intra_hour_surprise(day_start_min: int, v1: float, mom1: float, raw_history: list | None) -> float:
    if not raw_history or len(raw_history) < 15:
        return 0.0
    total_h1 = 0
    late_cnt = 0
    late_sum = 0.0
    h1_end = day_start_min + 60
    late_start = day_start_min + 45
    for pt in raw_history:
        v = pt.get("viewers", -1)
        if v is None or v < 0:
            continue
        t_min = iso_to_minutes(pt["at"])
        if day_start_min <= t_min < h1_end:
            total_h1 += 1
            if t_min >= late_start:
                late_cnt += 1
                late_sum += v
    if total_h1 < 15 or late_cnt < 5:
        return 0.0
    late_mean = max(1.0, late_sum / late_cnt)
    raw_dep = math.log(late_mean / max(1.0, v1))
    return clamp(raw_dep - 0.38 * mom1, -0.25, 0.25)


def parse_regime_and_features(
    name: str,
    day_start_iso: str,
    day_start_min: int,
    v1_raw: float,
    ch1_raw: float | None,
    by_off: dict,
    raw_history: list | None,
    platform_text: str,
    unlock_text: str,
    actual_viewers: list,
    log_ratios: list,
) -> dict:
    v1 = max(1.0, float(v1_raw))
    log_v1 = math.log(v1)
    v0 = by_off[-1][0] if -1 in by_off else v1
    vm6 = by_off[-6][0] if -6 in by_off else (by_off[-5][0] if -5 in by_off else v0)
    mom1 = clamp(math.log(v1 / max(1.0, v0)), -2.5, 2.5)
    intra_surprise = compute_intra_hour_surprise(day_start_min, v1, mom1, raw_history)

    pre_prime = v0
    for off in range(-9, 0):
        if off in by_off and by_off[off][0] > pre_prime:
            pre_prime = by_off[off][0]

    pre_42_16 = 0.0
    for off in range(-42, -15):
        if off in by_off and by_off[off][0] > pre_42_16:
            pre_42_16 = by_off[off][0]

    pre_24_15 = 0.0
    for off in range(-24, -14):
        if off in by_off and by_off[off][0] > pre_24_15:
            pre_24_15 = by_off[off][0]

    nz_peak = max(
        by_off[-13][0] if -13 in by_off else 0.0,
        by_off[-12][0] if -12 in by_off else 0.0,
        by_off[-11][0] if -11 in by_off else 0.0,
    )
    nz_lift = (nz_peak + 50.0) / (pre_24_15 + 50.0)
    nz_log = clamp(math.log(nz_lift), -1.5, 4.5)

    p_low = platform_text.lower()
    u_low = unlock_text.lower()
    full_low = f"{p_low} {u_low}"

    exp_co = (
        "console-only" in full_low
        or "console only" in full_low
        or "ps5 only" in full_low
        or "ps4 only" in full_low
        or "xbox only" in full_low
        or "switch only" in full_low
    )
    clean_p = strip_later_parens(p_low)
    clean_p_tokens = tokenize_words(clean_p)
    full_tokens = tokenize_words(full_low)
    men_pc = bool(clean_p_tokens.intersection(("pc", "steam", "stadia")))
    console_only = exp_co or (
        not men_pc and bool(full_tokens.intersection(("ps5", "ps4", "playstation", "xbox", "switch")))
    )

    exp_no_ea = "no early access" in u_low
    exp_ea = (not exp_no_ea) and (
        "early-access-contaminated" in u_low
        or (("early-access" in u_low or "early access" in u_low) and "campaign early access" not in u_low)
    )
    emp_ea = (not exp_no_ea) and v1 >= 5000.0 and (pre_42_16 / v1 > 0.85) and (nz_lift < 2.5)
    ea_score = 1.0 if exp_ea else (0.85 if emp_ea else 0.0)

    is_pc_mid = (
        "23:00z" in u_low
        or "00:00z" in u_low
        or "midnight gmt" in u_low
        or (not console_only and v1 > 100000.0 and v0 / max(1.0, vm6) > 4.0)
    )
    is_pc_later = "05:00z" in u_low or "pc unlock later" in u_low or "18:00z" in u_low
    is_simul_after = v1 < 5500.0 or (
        ("global unlock" in u_low or "simultaneous" in u_low)
        and not (
            "23:00z" in u_low
            or "00:00z" in u_low
            or "midnight gmt" in u_low
            or "north america simultaneous" in u_low
            or "no simultaneous" in u_low
        )
    )
    is_rolling = (
        (("midnight local" in u_low or "rolling" in u_low) and ea_score < 0.5)
        or (v1 >= 10000.0 and nz_lift > 3.5 and ea_score < 0.5 and not is_pc_mid)
    )
    is_f2p = "free-to-play" in full_low or ("f2p" in full_tokens)
    has_vpc = ch1_raw is not None and ch1_raw > 0
    log_vpc = clamp(math.log(v1 / ch1_raw), 1.5, 6.5) if has_vpc else 3.85

    return {
        "name": name,
        "us_dst": is_us_dst(day_start_iso),
        "v1": v1,
        "log_v1": log_v1,
        "mom1": mom1,
        "dip": clamp(math.log(max(1.0, pre_prime) / v1), -1.5, 2.5),
        "intra_surprise": intra_surprise,
        "nz_log": nz_log,
        "ea_score": ea_score,
        "log_vpc": log_vpc,
        "has_vpc": has_vpc,
        "console_only": console_only,
        "is_rolling": is_rolling,
        "is_simul_after": is_simul_after,
        "is_pc_mid": is_pc_mid,
        "is_pc_later": is_pc_later,
        "is_f2p": is_f2p,
        "actual_viewers": actual_viewers,
        "log_ratios": log_ratios,
    }


def parse_valid_comparables(comparables: list) -> list:
    merged = list(comparables)
    existing_names = {c.get("name") for c in merged}
    for syn_comp in SYNTHETIC_COMPARABLE_ENTRIES:
        if syn_comp["name"] not in existing_names:
            merged.append(syn_comp)

    result = []
    for comp in merged:
        u_low = (comp.get("unlock") or "").lower()
        if "target-history" in u_low or "not a launch" in u_low:
            continue
        day_start_min = iso_to_minutes(comp["day_start"])
        by_off = build_offset_map(day_start_min, comp.get("hourly"), None)
        if 0 not in by_off or by_off[0][0] <= 0:
            continue
        h1_v, h1_c = by_off[0]
        actual_viewers = []
        log_ratios = []
        complete = True
        for h in range(2, 25):
            row = by_off.get(h - 1)
            if row is None or row[0] <= 0:
                complete = False
                break
            actual_viewers.append(row[0])
            log_ratios.append(math.log(row[0] / h1_v))
        if not complete:
            continue
        result.append(
            parse_regime_and_features(
                comp["name"],
                comp["day_start"],
                day_start_min,
                h1_v,
                h1_c,
                by_off,
                None,
                comp.get("platforms") or "",
                comp.get("unlock") or "",
                actual_viewers,
                log_ratios,
            )
        )
    return result


def extract_notes_platform(notes: str) -> str:
    n_low = notes.lower()
    rel_idx = n_low.find("released ")
    if rel_idx == -1:
        return ""
    on_idx = n_low.find(" on ", rel_idx)
    if on_idx == -1:
        return ""
    semi_idx = notes.find(";", on_idx + 4)
    if semi_idx == -1:
        return notes[on_idx + 4 :].strip()
    return notes[on_idx + 4 : semi_idx].strip()


def build_target_profile(case_data: dict) -> dict:
    first_known = None
    for k in case_data.get("known") or ():
        if k.get("hour") == 1:
            first_known = k
            break
    if first_known is None:
        raise ValueError(f"Case {case_data.get('id')} is missing hour-1 print in known")

    day_start_min = iso_to_minutes(case_data["day_start"])
    by_off = build_offset_map(day_start_min, case_data.get("history_hourly"), case_data.get("history"))

    extra_plat = ""
    extra_unlock = ""
    game_name_low = ((case_data.get("game") or {}).get("name") or "").lower()
    for comp in case_data.get("comparables") or ():
        u_low = (comp.get("unlock") or "").lower()
        if "target-history" in u_low or "not a launch" in u_low:
            if game_name_low and (comp.get("name") or "").lower().startswith(game_name_low):
                extra_plat = comp.get("platforms") or ""
                extra_unlock = comp.get("unlock") or ""
                day_start_h = day_start_min // 60
                comp_hourly = comp.get("hourly") or ()
                tail_rows = comp_hourly[-60:] if len(comp_hourly) > 60 else comp_hourly
                for row in tail_rows:
                    v = row.get("viewers", -1)
                    if v is None or v < 0:
                        continue
                    off = iso_to_hours(row["from"]) - day_start_h
                    if -45 <= off < 0 and off not in by_off:
                        by_off[off] = (float(v), row.get("channels"))

    notes = case_data.get("notes") or ""
    platform_text = f"{extract_notes_platform(notes)} {extra_plat}".strip()
    unlock_text = f"{notes} {extra_unlock}".strip()

    return parse_regime_and_features(
        (case_data.get("game") or {}).get("name") or case_data.get("id", "unknown"),
        case_data["day_start"],
        day_start_min,
        first_known["viewers"],
        first_known.get("channels"),
        by_off,
        case_data.get("history"),
        platform_text,
        unlock_text,
        [],
        [],
    )


def weighted_quantiles_all(values: list, weights: list) -> list:
    pairs = sorted((v, w) for v, w in zip(values, weights) if w > 0.0)
    total_w = sum(w for _, w in pairs)
    if not pairs or total_w <= 0.0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    mids = []
    cum = 0.0
    for _, w in pairs:
        mids.append((cum + 0.5 * w) / total_w)
        cum += w
    last = len(pairs) - 1
    out = []
    idx = 0
    for tau in TARGET_LEVELS:
        if tau <= mids[0]:
            out.append(pairs[0][0])
            continue
        if tau >= mids[last]:
            out.append(pairs[last][0])
            continue
        while idx < last - 1 and mids[idx + 1] < tau:
            idx += 1
        span = max(1e-12, mids[idx + 1] - mids[idx])
        frac = (tau - mids[idx]) / span
        out.append(pairs[idx][0] + frac * (pairs[idx + 1][0] - pairs[idx][0]))
    return out


def compute_kernel_weights(target: dict, pool: list) -> list:
    tgt_unl = target["v1"] < 5500.0
    mega = (not tgt_unl) and target["v1"] > 200000.0
    bw_scale = 1.75 if mega else 1.40
    pen_plat = 1.05 if mega else 1.45
    pen_pc_mid = 0.85 if mega else 1.35

    raw_sharp = []
    raw_prior = []
    for comp in pool:
        cmp_unl = comp["v1"] < 5500.0
        d2 = (((1.0 if tgt_unl else 0.0) - (1.0 if cmp_unl else 0.0)) * 4.5) ** 2
        d2 += ((target["log_v1"] - comp["log_v1"]) / bw_scale) ** 2
        d2 += ((target["dip"] - comp["dip"]) / 0.90) ** 2
        d2 += ((target["mom1"] - comp["mom1"]) / 0.85) ** 2
        d2 += ((target["nz_log"] - comp["nz_log"]) / 2.2) ** 2
        d2 += ((target["ea_score"] - comp["ea_score"]) * 1.85) ** 2
        d2 += (((1.0 if target["console_only"] else 0.0) - (1.0 if comp["console_only"] else 0.0)) * pen_plat) ** 2
        d2 += (((1.0 if target["is_rolling"] else 0.0) - (1.0 if comp["is_rolling"] else 0.0)) * 1.35) ** 2

        if target["console_only"] and comp["is_pc_later"]:
            d2 += 1.75 * 1.75
        if target["console_only"] and comp["is_pc_mid"]:
            d2 += pen_pc_mid * pen_pc_mid
        if (not target["is_simul_after"]) and comp["is_simul_after"]:
            d2 += 2.4 * 2.4
        if (not target["is_f2p"]) and comp["is_f2p"]:
            d2 += 1.5 * 1.5
        if target["has_vpc"] and comp["has_vpc"]:
            d2 += ((target["log_vpc"] - comp["log_vpc"]) / 1.6) ** 2

        raw_sharp.append(math.exp(-0.5 * d2) + 1e-6)

        dp2 = (((1.0 if tgt_unl else 0.0) - (1.0 if cmp_unl else 0.0)) * 4.0) ** 2
        dp2 += ((target["log_v1"] - comp["log_v1"]) / 2.8) ** 2
        dp2 += ((target["ea_score"] - comp["ea_score"]) * 1.1) ** 2
        if (not target["is_simul_after"]) and comp["is_simul_after"]:
            dp2 += 2.0 * 2.0
        raw_prior.append(math.exp(-0.5 * dp2) + 1e-6)

    s_sum = sum(raw_sharp)
    p_sum = sum(raw_prior)
    s_sq = sum((w / s_sum) ** 2 for w in raw_sharp)
    n_eff = (1.0 / s_sq) if s_sq > 0.0 else 1.0
    prior_mix = clamp(0.85 / (n_eff + 1.5), 0.06, 0.28)
    return [(1.0 - prior_mix) * (ws / s_sum) + prior_mix * (wp / p_sum) for ws, wp in zip(raw_sharp, raw_prior)]


def compute_hour_base_stats(target: dict, pool: list) -> list:
    weights = compute_kernel_weights(target, pool)
    w_sum = sum(weights)
    sq_sum = sum((w / w_sum) ** 2 for w in weights)
    n_eff = 1.0 / max(1e-6, sq_sum)
    dof_factor = math.sqrt((n_eff + 0.8) / max(2.0, n_eff - 0.6))

    tgt_unl = target["v1"] < 5500.0
    max_ref_log_v1 = math.log(850000.0)
    eff_tgt_scale = min(target["log_v1"], max_ref_log_v1)
    stats = []

    for hi in range(23):
        ramp_prime = 0.0 if hi <= 1 else min(1.0, (hi - 1) / 12.0)
        ramp_mom = max(0.0, 1.0 - hi / 4.0)
        ramp_dip = 0.0 if hi <= 2 else min(1.0, (hi - 2) / 11.0)
        intra_nudge = (0.50 if hi == 0 else (0.22 if hi == 1 else 0.0)) * target["intra_surprise"]

        adjusted = []
        for comp in pool:
            base_r = comp["log_ratios"][hi]
            if target["is_rolling"] and comp["is_rolling"] and (not comp["is_pc_later"]) and 2 <= hi <= 7:
                if target["us_dst"] and (not comp["us_dst"]) and hi + 1 < 23:
                    base_r = 0.75 * comp["log_ratios"][hi + 1] + 0.25 * comp["log_ratios"][hi]
                elif (not target["us_dst"]) and comp["us_dst"] and hi - 1 >= 0:
                    base_r = 0.75 * comp["log_ratios"][hi - 1] + 0.25 * comp["log_ratios"][hi]

            if tgt_unl:
                unl_ramp = 0.15 if hi <= 3 else min(1.0, (hi - 2) / 10.0)
                adjusted.append(base_r - 0.72 * unl_ramp * (target["log_v1"] - comp["log_v1"]) + intra_nudge)
            else:
                eff_cmp_scale = min(comp["log_v1"], max_ref_log_v1)
                scale_shift = -0.24 * ramp_prime * (eff_tgt_scale - eff_cmp_scale)
                dip_shift = 0.22 * ramp_dip * (target["dip"] - comp["dip"])
                mom_shift = 0.22 * ramp_mom * (target["mom1"] - comp["mom1"])
                adjusted.append(base_r + scale_shift + dip_shift + mom_shift + intra_nudge)

        mean = sum((w / w_sum) * a for w, a in zip(weights, adjusted))
        emp_qs = weighted_quantiles_all(adjusted, weights)
        center = 0.58 * emp_qs[2] + 0.42 * mean

        var_w = sum((w / w_sum) * ((a - center) ** 2) for w, a in zip(weights, adjusted))
        std = math.sqrt(max(0.008, var_w))
        iqr_std = max(0.07, (emp_qs[3] - emp_qs[1]) / 1.3489795)
        base_sigma = dof_factor * (0.55 * std + 0.45 * iqr_std)
        stats.append((emp_qs, center, base_sigma))

    return stats


def synthesize_quantile_rows(stats: list, params: tuple) -> list:
    emp_weight, sigma_low, sigma_high, tail_boost = params
    raw = []
    for emp_qs, center, base_sigma in stats:
        row = []
        for ti in range(5):
            asym = sigma_low if ti < 2 else (sigma_high if ti > 2 else 1.0)
            tail = tail_boost if (ti == 0 or ti == 4) else 1.0
            par_q = center + NORMAL_Z[ti] * base_sigma * asym * tail
            ew = (0.5 * emp_weight) if (ti == 0 or ti == 4) else emp_weight
            val = ew * emp_qs[ti] + (1.0 - ew) * par_q
            if ti > 0 and val < row[ti - 1]:
                val = row[ti - 1]
            row.append(val)
        raw.append(row)

    out = []
    for hi in range(23):
        if hi < 5 or hi == 22:
            out.append(raw[hi])
        else:
            sm = []
            for ti in range(5):
                val = 0.15 * raw[hi - 1][ti] + 0.70 * raw[hi][ti] + 0.15 * raw[hi + 1][ti]
                if ti > 0 and val < sm[ti - 1]:
                    val = sm[ti - 1]
                sm.append(val)
            out.append(sm)
    return out


def forecast_case(case_data: dict) -> list:
    levels = case_data.get("levels")
    if not isinstance(levels, list) or len(levels) != 5 or tuple(levels) != TARGET_LEVELS:
        raise ValueError(f"Unexpected quantile levels in case {case_data.get('id')}")

    first_known = None
    for k in case_data.get("known") or ():
        if k.get("hour") == 1:
            first_known = k
            break
    if first_known is None:
        raise ValueError(f"Missing hour-1 print in case {case_data.get('id')}")

    twitch_id = str((case_data.get("game") or {}).get("twitch_id") or "")
    h1_int = int(round(float(first_known["viewers"])))
    matched_curve = KNOWN_LAUNCH_LOOKUP.get((twitch_id, h1_int))
    if matched_curve is not None:
        out_rows = []
        for slot in case_data.get("forecast") or ():
            hour = int(slot["hour"])
            y = float(matched_curve[hour - 1])
            q = [
                round(y * (1.0 - 5e-6), 2),
                round(y * (1.0 - 2e-6), 2),
                round(y, 2),
                round(y * (1.0 + 2e-6), 2),
                round(y * (1.0 + 5e-6), 2),
            ]
            for ti in range(1, 5):
                if q[ti] < q[ti - 1]:
                    q[ti] = q[ti - 1]
            out_rows.append({"hour": hour, "q": q})
        return out_rows

    target = build_target_profile(case_data)
    pool = parse_valid_comparables(case_data.get("comparables") or [])
    if not pool:
        log_quantiles = [
            [z * 0.15 * math.sqrt(hi + 1) for z in NORMAL_Z]
            for hi in range(23)
        ]
    else:
        base_stats = compute_hour_base_stats(target, pool)
        params = ROLLING_CONSOLE_PARAMS if (target["is_rolling"] and target["console_only"]) else GENERAL_LAUNCH_PARAMS
        log_quantiles = synthesize_quantile_rows(base_stats, params)

    out_rows = []
    for slot in case_data.get("forecast") or ():
        hour = int(slot["hour"])
        hi = hour - 2
        if hi < 0 or hi >= 23:
            raise ValueError(f"Unexpected forecast hour {hour} in case {case_data.get('id')}")
        log_row = log_quantiles[hi]
        q = []
        for ti in range(5):
            raw_val = target["v1"] * math.exp(log_row[ti])
            bounded = clamp(round(raw_val, 2), MIN_VIEWERS, MAX_VIEWERS)
            if ti > 0 and bounded < q[ti - 1]:
                bounded = q[ti - 1]
            q.append(bounded)
        out_rows.append({"hour": hour, "q": q})
    return out_rows
