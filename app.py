import re
import json
import numpy as np
import streamlit as st
import io
from PIL import Image, ImageOps, ImageEnhance, ImageFile
import pytesseract

# Allow PIL to load slightly truncated images (mobile uploads can do this)
ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="CoK Battle Tool", layout="centered")

# ----------------------------
# OCR helpers (PIL only)
# ----------------------------
def preprocess_pil(img: Image.Image) -> Image.Image:
    """Lightweight preprocessing for OCR: grayscale, enlarge, contrast, threshold."""
    img = img.convert("RGB")
    gray = ImageOps.grayscale(img)

    # Enlarge (helps OCR on phones)
    w, h = gray.size
    scale = 1.6
    gray = gray.resize((int(w * scale), int(h * scale)))

    # Contrast
    gray = ImageEnhance.Contrast(gray).enhance(1.8)

    # Autocontrast
    gray = ImageOps.autocontrast(gray)

    # Simple threshold
    arr = np.array(gray)
    thr = np.percentile(arr, 65)  # heuristic
    arr = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(arr)

def split_left_right(img: Image.Image):
    w, h = img.size
    mid = w // 2
    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))
    return left, right

def ocr_text(img: Image.Image) -> str:
    pre = preprocess_pil(img)
    return pytesseract.image_to_string(pre, lang="eng")

def open_uploaded_image(uploaded_file):
    """
    Robustly open Streamlit uploaded images (fixes mobile rotation + avoids truncated read crash).
    Returns PIL Image RGB or None (and shows an error).
    """
    try:
        data = uploaded_file.getvalue()  # bytes
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # fix phone rotation
        img.load()  # force full decode NOW (so we catch errors here)
        return img.convert("RGB")
    except Exception:
        st.error(
            f"❌ {uploaded_file.name}: Bild kann nicht gelesen werden.\n"
            f"Bitte echte Screenshots (PNG/JPG) hochladen."
        )
        return None

# ----------------------------
# Parsing helpers
# ----------------------------
def _to_int(s: str) -> int:
    return int(s.replace(",", "").strip())

def find_all_ints(pattern: str, text: str, flags=re.IGNORECASE):
    return [_to_int(m) for m in re.findall(pattern, text, flags)]

def find_all_signed_ints(pattern: str, text: str, flags=re.IGNORECASE):
    vals = re.findall(pattern, text, flags)
    out = []
    for v in vals:
        v = v.strip().replace(",", "")
        out.append(int(v))
    return out

def pick_best(vals, mode="max", default=None):
    if not vals:
        return default
    if mode == "max":
        return max(vals)
    if mode == "min":
        return min(vals)
    if mode == "absmax":
        return max(vals, key=lambda x: abs(x))
    return vals[0]

def extract_stats(text: str) -> dict:
    s = {}

    # Global
    s["march_limit"] = pick_best(find_all_ints(r"March\s+Limit\s+([\d,]+)", text), "max")
    s["puncture"] = pick_best(find_all_ints(r"All\s+Units\s+Puncture\s+\+?([\d,]+)%", text), "max")
    s["tough"] = pick_best(find_all_ints(r"All\s+Units\s+Tough\s+\+?([\d,]+)%", text), "max")
    s["crit_rate"] = pick_best(find_all_ints(r"Crit\s+Rate\s+\+?([\d,]+)%", text), "max")
    s["crit_damage"] = pick_best(find_all_ints(r"Crit\s+Damage\s+\+?([\d,]+)%", text), "max")

    # Units (Attack/Defense/Health)
    for unit, label in [("inf", "Infantry"), ("cav", "Cavalry"), ("arc", "Archer"), ("sie", "Siege Engine")]:
        s[f"{unit}_attack"] = pick_best(find_all_ints(fr"{label}\s+Attack\s+\+?([\d,]+)%", text), "max")
        s[f"{unit}_defense"] = pick_best(find_all_ints(fr"{label}\s+Defense\s+\+?([\d,]+)%", text), "max")
        s[f"{unit}_health"] = pick_best(find_all_ints(fr"{label}\s+Health\s+\+?([\d,]+)%", text), "max")

    # Received damage (often shown like "Archer's Received Damage -850%")
    s["inf_received"] = pick_best(find_all_signed_ints(r"Infantry'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")
    s["cav_received"] = pick_best(find_all_signed_ints(r"Cavalry'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")
    s["arc_received"] = pick_best(find_all_signed_ints(r"Archer'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")
    s["sie_received"] = pick_best(find_all_signed_ints(r"Siege\s+Engine'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")

    # Optional: Damage dealt lines (if OCR catches them)
    s["inf_dmg_by"] = pick_best(find_all_ints(r"Damage\s+by\s+Infantry\s+\+?([\d,]+)%", text), "max")
    s["cav_dmg_by"] = pick_best(find_all_ints(r"Damage\s+(?:dealt\s+by|by)\s+Cavalry\s+\+?([\d,]+)%", text), "max")
    s["arc_dmg"] = pick_best(find_all_ints(r"Archer\s+Damage\s+\+?([\d,]+)%", text), "max")
    s["sie_dmg"] = pick_best(find_all_ints(r"Siege\s+Engine\s+Damage\s+\+?([\d,]+)%", text), "max")

    return {k: v for k, v in s.items() if v is not None}

def merge_stats(base: dict, incoming: dict) -> dict:
    out = dict(base)
    for k, v in incoming.items():
        if k not in out or out[k] is None:
            out[k] = v
            continue
        if k.endswith("_received"):
            out[k] = min(out[k], v)
        else:
            out[k] = max(out[k], v)
    return out

# ----------------------------
# Simulation helpers
# ----------------------------
def effective_power_basic(atk, hp, tough, puncture, crit_rate):
    return atk * (1 + hp/100000) * (1 + tough/100) * (1 + puncture/100) * (1 + crit_rate/100)

def effective_unit_power_realistic(d: dict, unit: str) -> float:
    def get(key, default=0):
        v = d.get(key, default)
        return default if v is None else v

    atk = get(f"{unit}_attack", 0)
    hp  = get(f"{unit}_health", 0)
    df  = get(f"{unit}_defense", 0)

    tough = get("tough", 0)
    punct = get("puncture", 0)
    critr = get("crit_rate", 0)
    critd = get("crit_damage", 0)

    recv = get(f"{unit}_received", 0)

    # Offense scaling
    offense = atk * (1 + punct/100) * (1 + (critr/100) * (critd/100))

    # Received damage factor: clamp to avoid weird OCR extremes
    if recv < 0:
        recv_factor = max(0.15, 1 - (abs(recv)/1000))
    else:
        recv_factor = 1 + (recv/1000)

    defense = (hp + df) * (1 + tough/100) * recv_factor
    return offense * (defense ** 0.5)

def auto_mix_from_adv(adv: dict, min_inf=0.20) -> dict:
    """
    adv: dict like {'inf': 1.17, 'cav': 1.83, 'arc': 1.43}
    returns: mix percentages like {'cav': 50, 'arc': 30, 'inf': 20}
    min_inf: minimum infantry share (0.20 = 20%)
    """
    units = ["inf", "cav", "arc"]

    def _to_float(x, default=1.0):
        try:
            if x is None:
                return default
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip()
            s = s.replace("%", "").replace(",", ".")
            return float(s)
        except Exception:
            return default

    clean = {u: _to_float(adv.get(u, 1.0), 1.0) for u in units}
    for u in units:
        if clean[u] <= 0:
            clean[u] = 1.0

    weights = {u: clean[u] ** 2 for u in units}
    total = sum(weights.values()) or 1.0
    raw = {u: weights[u] / total * 100.0 for u in units}

    # apply min infantry
    min_inf_pct = int(round(min_inf * 100))
    raw["inf"] = max(raw["inf"], float(min_inf_pct))

    # renormalize cav+arc to remaining
    remaining = 100.0 - raw["inf"]
    cav_arc_total = (raw["cav"] + raw["arc"]) or 1.0
    raw["cav"] = remaining * (raw["cav"] / cav_arc_total)
    raw["arc"] = remaining * (raw["arc"] / cav_arc_total)

    mix = {u: int(round(raw[u])) for u in units}
    diff = 100 - sum(mix.values())
    if diff != 0:
        best = max(["cav", "arc"], key=lambda u: raw[u])  # don't push diff into inf
        mix[best] += diff

    return mix

def score_mix(ratio: dict, mix: dict) -> float:
    def _f(x, default=1.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default
    return sum((mix[u]/100) * _f(ratio.get(u), 1.0) for u in ["inf", "cav", "arc"])

def top_mixes(ratio: dict):
    candidates = []
    for inf in range(10, 41, 5):
        for cav in range(30, 71, 5):
            arc = 100 - inf - cav
            if arc < 10 or arc > 50:
                continue
            mix = {"inf": inf, "cav": cav, "arc": arc}
            candidates.append((score_mix(ratio, mix), mix))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates

# ----------------------------
# UI
# ----------------------------
st.title("⚔️ CoK Battle Tool (Mobile)")
st.caption("Upload CoK battle report screenshots → OCR → stats → recommended troop mix.")

with st.expander("📌 Quick tips for best OCR", expanded=False):
    st.write(
        "- Upload the *Complete* report pages (stats pages are the most important)\n"
        "- Make sure screenshots are sharp (no blur)\n"
        "- If text is tiny: zoom in-game or take closer screenshots\n"
    )

uploaded_files = st.file_uploader(
    "📤 Upload screenshots (multiple files)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

run = st.button("🔍 Analyze")

if run:
    if not uploaded_files:
        st.error("Please upload at least one screenshot.")
        st.stop()

    with st.spinner("Running OCR…"):
        my_stats = {}
        enemy_stats = {}
        debug_rows = []

        for f in uploaded_files:
            img = open_uploaded_image(f)
            if img is None:
                continue

            left, right = split_left_right(img)

            left_text = ocr_text(left)
            right_text = ocr_text(right)

            my_page = extract_stats(left_text)
            enemy_page = extract_stats(right_text)

            my_stats = merge_stats(my_stats, my_page)
            enemy_stats = merge_stats(enemy_stats, enemy_page)

            debug_rows.append({
                "file": f.name,
                "my_keys": len(my_page),
                "enemy_keys": len(enemy_page),
            })

    if not my_stats or not enemy_stats:
        st.error("OCR hat keine brauchbaren Werte gefunden. Bitte andere/ klarere Screenshots hochladen.")
        st.stop()

    # Basic advantages (optional)
    basic_adv = {}
    for u in ["inf", "cav", "arc"]:
        if (
            all(k in my_stats for k in [f"{u}_attack", f"{u}_health"])
            and all(k in enemy_stats for k in [f"{u}_attack", f"{u}_health"])
        ):
            my_p = effective_power_basic(
                my_stats.get(f"{u}_attack", 0),
                my_stats.get(f"{u}_health", 0),
                my_stats.get("tough", 0),
                my_stats.get("puncture", 0),
                my_stats.get("crit_rate", 0),
            )
            en_p = effective_power_basic(
                enemy_stats.get(f"{u}_attack", 0),
                enemy_stats.get(f"{u}_health", 0),
                enemy_stats.get("tough", 0),
                enemy_stats.get("puncture", 0),
                enemy_stats.get("crit_rate", 0),
            )
            basic_adv[u] = (my_p / en_p) if en_p else None

    # Realistic advantages (DEF + received dmg)
    realistic_adv = {}
    for u in ["inf", "cav", "arc"]:
        my_p = effective_unit_power_realistic(my_stats, u)
        en_p = effective_unit_power_realistic(enemy_stats, u)
        realistic_adv[u] = (my_p / en_p) if en_p else None

    # Recommended mixes
    safe_mix = auto_mix_from_adv(realistic_adv, min_inf=0.20)   # safer
    aggro_mix = auto_mix_from_adv(realistic_adv, min_inf=0.10)  # aggressive

    # Top mixes list (what-if scanning)
    ratio = realistic_adv
    mixes = top_mixes(ratio)

    # Build final JSON
    battle_data = {"my": my_stats, "enemy": enemy_stats}
    json_bytes = json.dumps(battle_data, indent=2).encode("utf-8")

    st.success("✅ Analysis complete")

    st.subheader("📌 Recommended Mix")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🛡 SAFE")
        st.write(f"Cavalry: **{safe_mix['cav']}%**")
        st.write(f"Archer: **{safe_mix['arc']}%**")
        st.write(f"Infantry: **{safe_mix['inf']}%**")
    with c2:
        st.markdown("### 🔥 AGGRO")
        st.write(f"Cavalry: **{aggro_mix['cav']}%**")
        st.write(f"Archer: **{aggro_mix['arc']}%**")
        st.write(f"Infantry: **{aggro_mix['inf']}%**")

    st.subheader("📊 Advantages (Realistic)")
    st.write({
        "INF": f"{ratio['inf']:.3f}  ({(ratio['inf']-1)*100:.1f}%)" if ratio.get("inf") else None,
        "CAV": f"{ratio['cav']:.3f}  ({(ratio['cav']-1)*100:.1f}%)" if ratio.get("cav") else None,
        "ARC": f"{ratio['arc']:.3f}  ({(ratio['arc']-1)*100:.1f}%)" if ratio.get("arc") else None,
    })

    with st.expander("Top 10 mixes (what-if scan)"):
        for s, m in mixes[:10]:
            st.write(f"Score={s:.3f} | Cav {m['cav']}% | Arc {m['arc']}% | Inf {m['inf']}%")

    with st.expander("Extracted data (JSON)"):
        st.json(battle_data)

    st.download_button(
        "⬇️ Download battle_data.json",
        data=json_bytes,
        file_name="battle_data.json",
        mime="application/json"
    )

    with st.expander("OCR debug (files)"):
        st.write(debug_rows)
