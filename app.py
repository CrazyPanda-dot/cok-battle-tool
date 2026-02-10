import re
import json
import numpy as np
import streamlit as st
import io
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageFile
import pytesseract

ImageFile.LOAD_TRUNCATED_IMAGES = True

st.set_page_config(page_title="CoK Battle Tool", layout="centered")

# ----------------------------
# OCR helpers (PIL only)
# ----------------------------
def open_uploaded_image(uploaded_file):
    """
    Robust read:
    - reads bytes safely
    - fixes phone rotation (EXIF)
    - forces full load
    """
    data = uploaded_file.getvalue()
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img.load()
        return img.convert("RGB")
    except Exception:
        st.error(
            f"❌ {uploaded_file.name}: Bild kann nicht gelesen werden.\n"
            f"Bitte echte Screenshots (PNG/JPG) hochladen."
        )
        return None


def preprocess_pil(img: Image.Image) -> Image.Image:
    """
    Strong-but-stable preprocessing for CoK battle report lists:
    - grayscale
    - enlarge
    - autocontrast
    - sharpen
    - threshold (percentile)
    """
    img = img.convert("RGB")
    gray = ImageOps.grayscale(img)

    # Enlarge (helps on phone screenshots)
    w, h = gray.size
    scale = 2.0
    gray = gray.resize((int(w * scale), int(h * scale)))

    # Contrast & autocontrast
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageOps.autocontrast(gray)

    # Sharpen slightly
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))

    # Threshold
    arr = np.array(gray)
    thr = np.percentile(arr, 62)  # tuned for beige background + dark text
    arr = (arr > thr).astype(np.uint8) * 255

    return Image.fromarray(arr)


def split_left_right(img: Image.Image):
    """
    Split report into left/right columns.
    Some screenshots have uneven margins; a tiny bias helps.
    """
    w, h = img.size
    mid = int(w * 0.50)  # 0.50 works well for your screenshots
    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))
    return left, right


def ocr_text(img: Image.Image) -> str:
    pre = preprocess_pil(img)
    # psm 6 = assume a uniform block of text; good for lists
    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(pre, lang="eng", config=config)


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
    """
    Extract the core stats from CoK report pages (your screenshots).
    We focus on the lines that reliably appear:
    - March Limit
    - All Units Puncture/Tough
    - Infantry/Cavalry/Archer Attack/Defense/Health
    - Received Damage lines
    - Crit Rate / Crit Damage (unit-specific)
    """
    s = {}

    # Global (your screenshots include these on "Unit Attributes" page)
    s["march_limit"] = pick_best(find_all_ints(r"March\s+Limit\s+([\d,]+)", text), "max")
    s["puncture"] = pick_best(find_all_ints(r"All\s+Units\s+Puncture\s+\+?([\d,]+)%", text), "max")
    s["tough"] = pick_best(find_all_ints(r"All\s+Units\s+Tough\s+\+?([\d,]+)%", text), "max")

    # Unit A/D/HP (your screenshots use "Health", not "HP" in these lines)
    for unit, label in [("inf", "Infantry"), ("cav", "Cavalry"), ("arc", "Archer"), ("sie", "Siege Engine")]:
        s[f"{unit}_attack"] = pick_best(find_all_ints(fr"{label}\s+Attack\s+\+?([\d,]+)%", text), "max")
        s[f"{unit}_defense"] = pick_best(find_all_ints(fr"{label}\s+Defense\s+\+?([\d,]+)%", text), "max")
        s[f"{unit}_health"] = pick_best(find_all_ints(fr"{label}\s+Health\s+\+?([\d,]+)%", text), "max")

    # Received damage (your screenshots show e.g. "Infantry's Received Damage -894%")
    s["inf_received"] = pick_best(find_all_signed_ints(r"Infantry'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")
    s["cav_received"] = pick_best(find_all_signed_ints(r"Cavalry'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")
    s["arc_received"] = pick_best(find_all_signed_ints(r"Archer'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")
    s["sie_received"] = pick_best(find_all_signed_ints(r"Siege\s+Engine'?s\s+Received\s+Damage\s+(-?[\d,]+)%", text), "min")

    # Unit Crit stats (your screenshots show "Cavalry Crit Rate +127%" etc.)
    for unit, label in [("inf", "Infantry"), ("cav", "Cavalry"), ("arc", "Archer"), ("sie", "Siege Engine")]:
        s[f"{unit}_crit_rate"] = pick_best(find_all_ints(fr"{label}\s+Crit\s+Rate\s+\+?([\d,]+)%", text), "max")
        s[f"{unit}_crit_damage"] = pick_best(find_all_ints(fr"{label}\s+Crit\s+Damage\s+\+?([\d,]+)%", text), "max")

    return {k: v for k, v in s.items() if v is not None}

def merge_stats(base: dict, incoming: dict) -> dict:
    """
    Merge OCR from multiple pages: keep best values.
    - received damage: take min (more negative usually better)
    - everything else: take max
    """
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
# Simulation / Mix helpers
# ----------------------------
def effective_unit_power(d: dict, unit: str) -> float:
    """
    Simple power model:
    - offense: attack scaled by crit
    - defense: (health + defense) scaled by received-damage reduction
    """
    atk = d.get(f"{unit}_attack", 0)
    hp  = d.get(f"{unit}_health", 0)
    df  = d.get(f"{unit}_defense", 0)

    # Unit crits (if missing, default 0)
    cr = d.get(f"{unit}_crit_rate", 0)
    cd = d.get(f"{unit}_crit_damage", 0)

    recv = d.get(f"{unit}_received", 0)

    # offense scaling
    offense = atk * (1 + (cr/100) * (cd/100))

    # received damage factor (clamp against OCR weirdness)
    # recv is negative in reports (e.g. -894)
    if recv < 0:
        recv_factor = max(0.15, 1 - (abs(recv)/1000))
    else:
        recv_factor = 1 + (recv/1000)

    defense = (hp + df) * recv_factor
    return offense * (defense ** 0.5)


def auto_mix_from_adv(adv: dict, min_inf: float = 0.20) -> dict:
    """
    adv: dict like {'inf': 1.17, 'cav': 1.83, 'arc': 1.43}
    min_inf: minimum infantry share (e.g. 0.20 = 20%)
    returns: mix percentages {'cav': 50, 'arc': 30, 'inf': 20}
    """
    units = ["inf", "cav", "arc"]

    def _to_float(x, default=1.0):
        try:
            if x is None:
                return default
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace("%", "").replace(",", ".")
            return float(s)
        except Exception:
            return default

    clean = {u: _to_float(adv.get(u, 1.0), 1.0) for u in units}
    for u in units:
        if clean[u] <= 0:
            clean[u] = 1.0

    # emphasize stronger advantages
    weights = {u: clean[u] ** 2 for u in units}
    total = sum(weights.values()) or 1.0

    raw = {u: weights[u] / total for u in units}

    # enforce min infantry
    min_inf = max(0.0, min(0.6, float(min_inf)))
    if raw["inf"] < min_inf:
        deficit = min_inf - raw["inf"]
        raw["inf"] = min_inf

        # take deficit from the worse of cav/arc proportional to their weights
        pool_units = ["cav", "arc"]
        pool_sum = raw["cav"] + raw["arc"] or 1.0
        raw["cav"] -= deficit * (raw["cav"] / pool_sum)
        raw["arc"] -= deficit * (raw["arc"] / pool_sum)

    # convert to %
    mix = {u: int(round(raw[u] * 100)) for u in units}
    diff = 100 - sum(mix.values())
    if diff != 0:
        best = max(units, key=lambda u: raw[u])
        mix[best] += diff

    return mix


def score_mix(ratio: dict, mix: dict) -> float:
    return sum((mix[u]/100) * (ratio.get(u, 1.0) or 1.0) for u in ["inf", "cav", "arc"])

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
        "- Nimm Screenshots von **Unit Attributes** (All Units Puncture/Tough + March Limit) und den langen Stats-Seiten\n"
        "- Kein Scroll-Blur, keine Bewegung, am besten **2–5 Screens** pro Kampf\n"
        "- Wenn es nicht matched: OCR-Debug unten öffnen und schauen, was gelesen wurde\n"
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
        ocr_debug = []  # store OCR text per file

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

            ocr_debug.append({
                "file": f.name,
                "left_text": left_text,
                "right_text": right_text,
            })

    # If absolutely nothing extracted, show helpful message + OCR debug
    if len(my_stats) == 0 and len(enemy_stats) == 0:
        st.error("OCR hat keine brauchbaren Werte gefunden. Bitte OCR-Debug unten öffnen und/oder klarere Screenshots hochladen.")
        with st.expander("🔎 OCR Debug (raw text)"):
            for item in ocr_debug[:10]:
                st.markdown(f"### {item['file']}")
                st.text("---- LEFT (me) ----")
                st.text(item["left_text"][:4000])
                st.text("---- RIGHT (enemy) ----")
                st.text(item["right_text"][:4000])
        st.stop()

    # Advantages (Realistic)
    realistic_adv = {}
    for u in ["inf", "cav", "arc"]:
        my_p = effective_unit_power(my_stats, u)
        en_p = effective_unit_power(enemy_stats, u)
        realistic_adv[u] = (my_p / en_p) if en_p else None

    # Recommended mixes
    safe_mix = auto_mix_from_adv(realistic_adv, min_inf=0.20)
    aggro_mix = auto_mix_from_adv(realistic_adv, min_inf=0.10)

    # Top mixes list (what-if scanning)
    mixes = top_mixes(realistic_adv)

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
        "INF": f"{realistic_adv['inf']:.3f} ({(realistic_adv['inf']-1)*100:.1f}%)" if realistic_adv["inf"] else None,
        "CAV": f"{realistic_adv['cav']:.3f} ({(realistic_adv['cav']-1)*100:.1f}%)" if realistic_adv["cav"] else None,
        "ARC": f"{realistic_adv['arc']:.3f} ({(realistic_adv['arc']-1)*100:.1f}%)" if realistic_adv["arc"] else None,
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

    with st.expander("🔎 OCR Debug (raw text)"):
        for item in ocr_debug[:10]:
            st.markdown(f"### {item['file']}")
            st.text("---- LEFT (me) ----")
            st.text(item["left_text"][:4000])
            st.text("---- RIGHT (enemy) ----")
            st.text(item["right_text"][:4000])
