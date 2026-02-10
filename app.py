import json
import math
import streamlit as st
import pandas as pd

st.set_page_config(page_title="CoK Battle Simulator (Manual + T17/T18)", layout="wide")


# ----------------------------
# Helpers
# ----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ensure_defaults_stats(d: dict) -> dict:
    defaults = {
        "march_limit": 547000,
        "puncture": 0,
        "tough": 0,
        "crit_rate": 0,
        "crit_damage": 0,
        "inf_attack": 0, "inf_defense": 0, "inf_health": 0, "inf_received": 0,
        "cav_attack": 0, "cav_defense": 0, "cav_health": 0, "cav_received": 0,
        "arc_attack": 0, "arc_defense": 0, "arc_health": 0, "arc_received": 0,
        "sie_attack": 0, "sie_defense": 0, "sie_health": 0, "sie_received": 0,
    }
    out = dict(defaults)
    out.update(d or {})
    return out


def normalize_to_limit(counts: dict, limit: int) -> dict:
    total = sum(max(0, int(v)) for v in counts.values())
    if total <= 0:
        return counts
    factor = limit / total
    scaled = {k: int(round(int(v) * factor)) for k, v in counts.items()}

    diff = limit - sum(scaled.values())
    if diff != 0:
        biggest = max(scaled.keys(), key=lambda k: scaled[k])
        scaled[biggest] += diff
    return scaled


def effective_unit_power_realistic(d: dict, unit: str) -> float:
    """
    unit in {"inf","cav","arc","sie"}
    Uses:
      unit_attack, unit_defense, unit_health, unit_received
      global: tough, puncture, crit_rate, crit_damage
    Returns a single "per-soldier power index" (heuristic).
    """
    def get(key, default=0.0):
        v = d.get(key, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    atk = get(f"{unit}_attack", 0)
    hp  = get(f"{unit}_health", 0)
    df  = get(f"{unit}_defense", 0)

    tough = get("tough", 0)
    punct = get("puncture", 0)
    critr = get("crit_rate", 0)
    critd = get("crit_damage", 0)

    recv = get(f"{unit}_received", 0)  # negative usually better

    offense = atk * (1 + punct / 100.0) * (1 + (critr / 100.0) * (critd / 100.0))

    if recv < 0:
        recv_factor = 1 - (abs(recv) / 1000.0)
        recv_factor = clamp(recv_factor, 0.15, 1.50)
    else:
        recv_factor = 1 + (recv / 1000.0)
        recv_factor = clamp(recv_factor, 0.15, 2.50)

    defense = (hp + df) * (1 + tough / 100.0) * recv_factor
    return offense * math.sqrt(max(defense, 1e-9))


def ratio_to_outcome(r: float) -> str:
    if r >= 1.25:
        return "✅ Sehr klarer Vorteil"
    if r >= 1.10:
        return "✅ Vorteil"
    if r >= 0.95:
        return "⚖️ Ausgeglichen / RNG / Setup"
    if r >= 0.80:
        return "⚠️ Nachteil"
    return "❌ Klarer Nachteil"


def df_to_roster(df: pd.DataFrame) -> list[dict]:
    """
    Ensure columns exist and types are correct.
    """
    needed = ["troop", "tier", "category", "modifier"]
    for c in needed:
        if c not in df.columns:
            df[c] = "" if c in ["troop", "tier", "category"] else 1.0

    out = []
    for _, r in df.iterrows():
        troop = str(r["troop"]).strip()
        if not troop:
            continue
        tier = str(r["tier"]).strip()
        cat = str(r["category"]).strip().lower()
        if cat not in ["inf", "cav", "arc", "sie"]:
            cat = "cav"
        try:
            mod = float(r["modifier"])
        except Exception:
            mod = 1.0
        mod = clamp(mod, 0.70, 1.40)
        out.append({"troop": troop, "tier": tier, "category": cat, "modifier": mod})
    return out


def roster_to_df(roster: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(roster, columns=["troop", "tier", "category", "modifier"])


# ----------------------------
# Defaults (your troops)
# IMPORTANT: category is a starting guess; you can change it in the UI.
# ----------------------------
DEFAULT_ROSTER = [
    {"troop": "Cannonback Ravager",        "tier": "T18", "category": "sie", "modifier": 1.00},
    {"troop": "Frenzied Undead Archer",    "tier": "T18", "category": "arc", "modifier": 1.00},
    {"troop": "Sabertooth",               "tier": "T18", "category": "cav", "modifier": 1.00},
    {"troop": "Griffinvanguard",          "tier": "T18", "category": "cav", "modifier": 1.02},
    {"troop": "Frostborne Guardian",      "tier": "T18", "category": "inf", "modifier": 1.02},
    {"troop": "Yeti Flayer",              "tier": "T17", "category": "cav", "modifier": 0.98},
    {"troop": "Warbull Stampeder",        "tier": "T17", "category": "cav", "modifier": 0.98},
    {"troop": "Froststeel Lancer",        "tier": "T17", "category": "cav", "modifier": 0.98},
    {"troop": "Froststeel Shieldbreaker", "tier": "T17", "category": "inf", "modifier": 0.98},
    {"troop": "Forsaked Archon",          "tier": "T18", "category": "inf", "modifier": 1.00},
    {"troop": "Bloodraven Bolt Reaper",   "tier": "T18", "category": "arc", "modifier": 1.00},
]


# ----------------------------
# Session state init
# ----------------------------
if "my_stats" not in st.session_state:
    st.session_state.my_stats = ensure_defaults_stats({
        "march_limit": 547000,
        "puncture": 313,
        "tough": 309,
        "crit_rate": 127,
        "crit_damage": 356,
        "inf_attack": 70330, "inf_defense": 66929, "inf_health": 67977, "inf_received": -894,
        "cav_attack": 82055, "cav_defense": 76380, "cav_health": 77814, "cav_received": -1033,
        "arc_attack": 68537, "arc_defense": 66807, "arc_health": 67416, "arc_received": -850,
        "sie_attack": 68563, "sie_defense": 66799, "sie_health": 67393, "sie_received": -851,
    })

if "enemy_stats" not in st.session_state:
    st.session_state.enemy_stats = ensure_defaults_stats({
        "march_limit": 508775,
        "puncture": 272,
        "tough": 255,
        "crit_rate": 137,
        "crit_damage": 336,
        "inf_attack": 65519, "inf_defense": 55686, "inf_health": 61024, "inf_received": -772,
        "cav_attack": 76329, "cav_defense": 64083, "cav_health": 68618, "cav_received": -927,
        "arc_attack": 65216, "arc_defense": 55451, "arc_health": 60019, "arc_received": -756,
        "sie_attack": 65348, "sie_defense": 55589, "sie_health": 60017, "sie_received": -785,
    })

if "roster" not in st.session_state:
    st.session_state.roster = DEFAULT_ROSTER

if "my_counts" not in st.session_state:
    # start with a cav-heavy template across YOUR troops
    initial = {r["troop"]: 0 for r in st.session_state.roster}
    # rough default distribution:
    for r in st.session_state.roster:
        if r["category"] == "cav":
            initial[r["troop"]] = 70000
        elif r["category"] == "inf":
            initial[r["troop"]] = 40000
        elif r["category"] == "arc":
            initial[r["troop"]] = 35000
        else:
            initial[r["troop"]] = 5000
    st.session_state.my_counts = normalize_to_limit(initial, int(st.session_state.my_stats["march_limit"]))

if "saved_configs" not in st.session_state:
    st.session_state.saved_configs = []


# ----------------------------
# UI
# ----------------------------
st.title("⚔️ CoK Battle Simulator (Manual Stats + T17/T18 Roster)")
st.caption("Deine Stats einmalig speichern → Gegner-Stats eingeben → Marsch mit deinen T17/T18 Truppen variieren (Griffins/Sabertooth etc.) → Ergebnis vergleichen.")


# ---- Stats panes
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🧍 Meine Stats (einmalig)")
    with st.expander("Meine Stats bearbeiten", expanded=True):
        ms = dict(st.session_state.my_stats)

        ms["march_limit"] = st.number_input("March Limit (du)", min_value=1, step=1000, value=int(ms["march_limit"]))

        st.markdown("**Global**")
        ms["puncture"] = st.number_input("All Units Puncture (%)", step=1, value=int(ms["puncture"]))
        ms["tough"] = st.number_input("All Units Tough (%)", step=1, value=int(ms["tough"]))
        ms["crit_rate"] = st.number_input("Crit Rate (%)", step=1, value=int(ms["crit_rate"]))
        ms["crit_damage"] = st.number_input("Crit Damage (%)", step=1, value=int(ms["crit_damage"]))

        st.markdown("**Infantry**")
        ms["inf_attack"] = st.number_input("Inf Attack (%)", step=1, value=int(ms["inf_attack"]))
        ms["inf_defense"] = st.number_input("Inf Defense (%)", step=1, value=int(ms["inf_defense"]))
        ms["inf_health"] = st.number_input("Inf Health (%)", step=1, value=int(ms["inf_health"]))
        ms["inf_received"] = st.number_input("Inf Received Damage (%) (negativ ist gut)", step=1, value=int(ms["inf_received"]))

        st.markdown("**Cavalry**")
        ms["cav_attack"] = st.number_input("Cav Attack (%)", step=1, value=int(ms["cav_attack"]))
        ms["cav_defense"] = st.number_input("Cav Defense (%)", step=1, value=int(ms["cav_defense"]))
        ms["cav_health"] = st.number_input("Cav Health (%)", step=1, value=int(ms["cav_health"]))
        ms["cav_received"] = st.number_input("Cav Received Damage (%) (negativ ist gut)", step=1, value=int(ms["cav_received"]))

        st.markdown("**Archer**")
        ms["arc_attack"] = st.number_input("Arc Attack (%)", step=1, value=int(ms["arc_attack"]))
        ms["arc_defense"] = st.number_input("Arc Defense (%)", step=1, value=int(ms["arc_defense"]))
        ms["arc_health"] = st.number_input("Arc Health (%)", step=1, value=int(ms["arc_health"]))
        ms["arc_received"] = st.number_input("Arc Received Damage (%) (negativ ist gut)", step=1, value=int(ms["arc_received"]))

        st.markdown("**Siege**")
        ms["sie_attack"] = st.number_input("Sie Attack (%)", step=1, value=int(ms["sie_attack"]))
        ms["sie_defense"] = st.number_input("Sie Defense (%)", step=1, value=int(ms["sie_defense"]))
        ms["sie_health"] = st.number_input("Sie Health (%)", step=1, value=int(ms["sie_health"]))
        ms["sie_received"] = st.number_input("Sie Received Damage (%) (negativ ist gut)", step=1, value=int(ms["sie_received"]))

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💾 Meine Stats speichern"):
                st.session_state.my_stats = ensure_defaults_stats(ms)
                st.session_state.my_counts = normalize_to_limit(st.session_state.my_counts, int(st.session_state.my_stats["march_limit"]))
                st.success("Gespeichert ✅")
        with c2:
            my_json = json.dumps(st.session_state.my_stats, indent=2).encode("utf-8")
            st.download_button("⬇️ Meine Stats JSON", data=my_json, file_name="my_stats.json", mime="application/json")
        with c3:
            up = st.file_uploader("⬆️ Meine Stats laden (JSON)", type=["json"], label_visibility="collapsed")
            if up is not None:
                try:
                    loaded = json.loads(up.getvalue().decode("utf-8"))
                    st.session_state.my_stats = ensure_defaults_stats(loaded)
                    st.session_state.my_counts = normalize_to_limit(st.session_state.my_counts, int(st.session_state.my_stats["march_limit"]))
                    st.success("JSON geladen ✅")
                except Exception as e:
                    st.error(f"JSON konnte nicht gelesen werden: {e}")

with col_right:
    st.subheader("👤 Gegner Stats (pro Fight)")
    with st.expander("Gegner-Stats bearbeiten", expanded=True):
        es = dict(st.session_state.enemy_stats)

        es["march_limit"] = st.number_input("March Limit (Gegner)", min_value=1, step=1000, value=int(es["march_limit"]))

        st.markdown("**Global**")
        es["puncture"] = st.number_input("Enemy All Units Puncture (%)", step=1, value=int(es["puncture"]))
        es["tough"] = st.number_input("Enemy All Units Tough (%)", step=1, value=int(es["tough"]))
        es["crit_rate"] = st.number_input("Enemy Crit Rate (%)", step=1, value=int(es["crit_rate"]))
        es["crit_damage"] = st.number_input("Enemy Crit Damage (%)", step=1, value=int(es["crit_damage"]))

        st.markdown("**Infantry**")
        es["inf_attack"] = st.number_input("Enemy Inf Attack (%)", step=1, value=int(es["inf_attack"]))
        es["inf_defense"] = st.number_input("Enemy Inf Defense (%)", step=1, value=int(es["inf_defense"]))
        es["inf_health"] = st.number_input("Enemy Inf Health (%)", step=1, value=int(es["inf_health"]))
        es["inf_received"] = st.number_input("Enemy Inf Received Damage (%)", step=1, value=int(es["inf_received"]))

        st.markdown("**Cavalry**")
        es["cav_attack"] = st.number_input("Enemy Cav Attack (%)", step=1, value=int(es["cav_attack"]))
        es["cav_defense"] = st.number_input("Enemy Cav Defense (%)", step=1, value=int(es["cav_defense"]))
        es["cav_health"] = st.number_input("Enemy Cav Health (%)", step=1, value=int(es["cav_health"]))
        es["cav_received"] = st.number_input("Enemy Cav Received Damage (%)", step=1, value=int(es["cav_received"]))

        st.markdown("**Archer**")
        es["arc_attack"] = st.number_input("Enemy Arc Attack (%)", step=1, value=int(es["arc_attack"]))
        es["arc_defense"] = st.number_input("Enemy Arc Defense (%)", step=1, value=int(es["arc_defense"]))
        es["arc_health"] = st.number_input("Enemy Arc Health (%)", step=1, value=int(es["arc_health"]))
        es["arc_received"] = st.number_input("Enemy Arc Received Damage (%)", step=1, value=int(es["arc_received"]))

        st.markdown("**Siege**")
        es["sie_attack"] = st.number_input("Enemy Sie Attack (%)", step=1, value=int(es["sie_attack"]))
        es["sie_defense"] = st.number_input("Enemy Sie Defense (%)", step=1, value=int(es["sie_defense"]))
        es["sie_health"] = st.number_input("Enemy Sie Health (%)", step=1, value=int(es["sie_health"]))
        es["sie_received"] = st.number_input("Enemy Sie Received Damage (%)", step=1, value=int(es["sie_received"]))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Gegner-Stats speichern"):
                st.session_state.enemy_stats = ensure_defaults_stats(es)
                st.success("Gespeichert ✅")
        with c2:
            en_json = json.dumps(st.session_state.enemy_stats, indent=2).encode("utf-8")
            st.download_button("⬇️ Gegner Stats JSON", data=en_json, file_name="enemy_stats.json", mime="application/json")


st.divider()

# ---- Roster editor
st.subheader("🧩 Deine Truppenliste (T17/T18) – einmal einstellen")
st.caption("Kategorie = welche Stat-Gruppe genutzt wird (inf/cav/arc/sie). Modifier = Feintuning pro Einheit (1.00 = neutral).")

roster_df = roster_to_df(st.session_state.roster)

edited = st.data_editor(
    roster_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "troop": st.column_config.TextColumn("Troop", required=True),
        "tier": st.column_config.TextColumn("Tier", help="z.B. T17/T18"),
        "category": st.column_config.SelectboxColumn("Category", options=["inf", "cav", "arc", "sie"]),
        "modifier": st.column_config.NumberColumn("Modifier", min_value=0.70, max_value=1.40, step=0.01),
    },
    key="roster_editor"
)

roster = df_to_roster(edited)
st.session_state.roster = roster

cA, cB, cC = st.columns([1, 1, 2])
with cA:
    roster_json = json.dumps(st.session_state.roster, indent=2).encode("utf-8")
    st.download_button("⬇️ Roster JSON", data=roster_json, file_name="roster.json", mime="application/json")
with cB:
    up_roster = st.file_uploader("⬆️ Roster laden (JSON)", type=["json"], label_visibility="collapsed")
    if up_roster is not None:
        try:
            loaded = json.loads(up_roster.getvalue().decode("utf-8"))
            # basic validation
            if isinstance(loaded, list):
                st.session_state.roster = df_to_roster(pd.DataFrame(loaded))
                # ensure counts contain all troops
                cur = dict(st.session_state.my_counts)
                for r in st.session_state.roster:
                    cur.setdefault(r["troop"], 0)
                # remove counts for deleted troops
                keep = {r["troop"] for r in st.session_state.roster}
                cur = {k: v for k, v in cur.items() if k in keep}
                st.session_state.my_counts = normalize_to_limit(cur, int(st.session_state.my_stats["march_limit"]))
                st.success("Roster geladen ✅")
                st.rerun()
            else:
                st.error("Roster JSON muss eine Liste sein.")
        except Exception as e:
            st.error(f"Roster JSON Fehler: {e}")
with cC:
    if st.button("↩️ Roster Reset (Default)"):
        st.session_state.roster = DEFAULT_ROSTER
        cur = {r["troop"]: 0 for r in DEFAULT_ROSTER}
        # set some defaults
        for r in DEFAULT_ROSTER:
            if r["category"] == "cav":
                cur[r["troop"]] = 70000
            elif r["category"] == "inf":
                cur[r["troop"]] = 40000
            elif r["category"] == "arc":
                cur[r["troop"]] = 35000
            else:
                cur[r["troop"]] = 5000
        st.session_state.my_counts = normalize_to_limit(cur, int(st.session_state.my_stats["march_limit"]))
        st.success("Reset ✅")
        st.rerun()

st.divider()

# ---- Troop counts
st.subheader("🧪 What-If: Marsch konfigurieren")
ms = st.session_state.my_stats
es = st.session_state.enemy_stats
limit = int(ms["march_limit"])

# ensure counts keys match roster
counts = dict(st.session_state.my_counts)
keep = {r["troop"] for r in st.session_state.roster}
counts = {k: v for k, v in counts.items() if k in keep}
for r in st.session_state.roster:
    counts.setdefault(r["troop"], 0)

st.caption(f"Ziel: **{limit:,}** Truppen (dein March Limit).")

cols = st.columns(3)
for i, r in enumerate(st.session_state.roster):
    with cols[i % 3]:
        counts[r["troop"]] = st.number_input(
            f"{r['tier']} {r['troop']}  [{r['category']}]",
            min_value=0,
            step=1000,
            value=int(counts.get(r["troop"], 0)),
            help="Anzahl dieser Einheit in deinem Marsch"
        )

total = sum(int(v) for v in counts.values())

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    st.metric("Total Troops", f"{total:,}")
with c2:
    st.metric("Diff zu Limit", f"{(limit - total):,}")
with c3:
    if st.button("↔️ Auf March Limit normieren"):
        counts = normalize_to_limit(counts, limit)
        st.session_state.my_counts = counts
        st.rerun()
with c4:
    if total != limit:
        st.warning("Dein Marsch ist nicht exakt auf dem March Limit. (Normieren macht Vergleiche sauberer.)")

st.session_state.my_counts = counts

# ---- Power calc
my_base = {u: effective_unit_power_realistic(ms, u) for u in ["inf", "cav", "arc", "sie"]}
en_base = {u: effective_unit_power_realistic(es, u) for u in ["inf", "cav", "arc", "sie"]}

my_comp_power = 0.0
my_cat_counts = {"inf": 0, "cav": 0, "arc": 0, "sie": 0}

for r in st.session_state.roster:
    troop = r["troop"]
    cat = r["category"]
    mod = float(r["modifier"])
    n = int(counts.get(troop, 0))
    my_cat_counts[cat] += n
    my_comp_power += n * my_base[cat] * mod

# Enemy mix (adjustable)
st.divider()
st.subheader("🎛 Gegner-Marschannahme (Mix)")

st.caption("Wenn du die gegnerische Truppenverteilung ungefähr kennst, stell sie hier ein. Summe wird automatisch normalisiert.")
em1, em2, em3, em4 = st.columns(4)
with em1:
    infp = st.slider("Enemy % Inf", 0, 100, 20, 1)
with em2:
    cavp = st.slider("Enemy % Cav", 0, 100, 50, 1)
with em3:
    arcp = st.slider("Enemy % Arc", 0, 100, 25, 1)
with em4:
    siep = st.slider("Enemy % Sie", 0, 100, 5, 1)

s = infp + cavp + arcp + siep
if s <= 0:
    st.error("Gegner-Mix darf nicht 0 sein.")
    enemy_mix = {"inf": 0.25, "cav": 0.50, "arc": 0.20, "sie": 0.05}
else:
    enemy_mix = {"inf": infp / s, "cav": cavp / s, "arc": arcp / s, "sie": siep / s}

en_limit = int(es["march_limit"])
en_comp_power = 0.0
for cat in ["inf", "cav", "arc", "sie"]:
    en_comp_power += (en_limit * enemy_mix[cat]) * en_base[cat]

ratio = (my_comp_power / en_comp_power) if en_comp_power > 0 else 0.0

# ---- Results
st.divider()
res1, res2, res3 = st.columns([1.2, 1.2, 2])

with res1:
    st.subheader("📊 Ergebnis")
    st.metric("Power Ratio (Du / Gegner)", f"{ratio:.3f}", delta=f"{(ratio - 1) * 100:+.1f}%")
    st.write(ratio_to_outcome(ratio))
    st.caption("Heuristik: Stats → Kategorie-Power → Troop-Modifier → Gegner-Mix.")

with res2:
    st.subheader("🧱 Deine Kategorien im Marsch")
    df_cat = pd.DataFrame([
        {"Kategorie": "Inf", "Count": my_cat_counts["inf"]},
        {"Kategorie": "Cav", "Count": my_cat_counts["cav"]},
        {"Kategorie": "Arc", "Count": my_cat_counts["arc"]},
        {"Kategorie": "Sie", "Count": my_cat_counts["sie"]},
    ])
    df_cat["%"] = (df_cat["Count"] / max(sum(df_cat["Count"]), 1) * 100).round(1)
    st.dataframe(df_cat, use_container_width=True, hide_index=True)

with res3:
    st.subheader("🧾 Basis-Power (pro Kategorie)")
    df_pow = pd.DataFrame([
        {"Side": "You", "Unit": "Inf", "PowerIndex": my_base["inf"]},
        {"Side": "You", "Unit": "Cav", "PowerIndex": my_base["cav"]},
        {"Side": "You", "Unit": "Arc", "PowerIndex": my_base["arc"]},
        {"Side": "You", "Unit": "Sie", "PowerIndex": my_base["sie"]},
        {"Side": "Enemy", "Unit": "Inf", "PowerIndex": en_base["inf"]},
        {"Side": "Enemy", "Unit": "Cav", "PowerIndex": en_base["cav"]},
        {"Side": "Enemy", "Unit": "Arc", "PowerIndex": en_base["arc"]},
        {"Side": "Enemy", "Unit": "Sie", "PowerIndex": en_base["sie"]},
    ])
    st.dataframe(df_pow, use_container_width=True, hide_index=True)


# ---- Save/compare configs
st.divider()
st.subheader("🧾 Konfiguration speichern & vergleichen")

cfg_name = st.text_input("Name für diese Konfiguration (z. B. 'Mehr Griffinvanguard', 'Sabertooth heavy')", value="")

b1, b2, b3 = st.columns([1, 1, 3])

with b1:
    if st.button("➕ Save Config"):
        row = {
            "name": cfg_name.strip() or f"Config {len(st.session_state.saved_configs) + 1}",
            "march_limit": int(ms["march_limit"]),
            "total": int(total),
            "inf": int(my_cat_counts["inf"]),
            "cav": int(my_cat_counts["cav"]),
            "arc": int(my_cat_counts["arc"]),
            "sie": int(my_cat_counts["sie"]),
            "enemy_mix": dict(enemy_mix),
            "power_ratio": float(ratio),
            "troops": dict(counts),
            "roster": st.session_state.roster,
            "my_stats": st.session_state.my_stats,
            "enemy_stats": st.session_state.enemy_stats,
        }
        st.session_state.saved_configs.append(row)
        st.success("Gespeichert ✅")

with b2:
    if st.button("🗑️ Clear Saved"):
        st.session_state.saved_configs = []
        st.info("Gelöscht.")

with b3:
    st.caption("Tipp: Speichere 4–6 Varianten (z.B. Griffinvanguard hoch/runter) und sortiere nach Ratio.")

if st.session_state.saved_configs:
    df = pd.DataFrame([{
        "Name": r["name"],
        "Total": r["total"],
        "Inf": r["inf"],
        "Cav": r["cav"],
        "Arc": r["arc"],
        "Sie": r["sie"],
        "Ratio": round(r["power_ratio"], 3),
        "Δ%": round((r["power_ratio"] - 1) * 100, 1),
    } for r in st.session_state.saved_configs]).sort_values("Ratio", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    saved_json = json.dumps(st.session_state.saved_configs, indent=2).encode("utf-8")
    st.download_button("⬇️ Download saved_configs.json", data=saved_json, file_name="saved_configs.json", mime="application/json")

    pick = st.selectbox("Details anzeigen:", [r["name"] for r in st.session_state.saved_configs])
    chosen = next(r for r in st.session_state.saved_configs if r["name"] == pick)
    with st.expander("Troop Breakdown (Details)", expanded=False):
        st.json(chosen["troops"])
else:
    st.info("Noch keine gespeicherten Konfigurationen.")
