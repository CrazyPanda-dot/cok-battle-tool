# app.py
import re
import json
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageFile
import pytesseract

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------------------
# Page
# --------------------------------------------------
st.set_page_config(
    page_title="CoK Battle Tool",
    layout="centered"
)

st.title("⚔️ CoK Battle Tool (Mobile)")
st.caption("Upload CoK battle report screenshots → OCR → stats → recommended troop mix")

# --------------------------------------------------
# OCR PREPROCESSING (VERY AGGRESSIVE – mobile UI)
# --------------------------------------------------
def preprocess(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")                      # grayscale
    img = img.resize((img.width * 2, img.height * 2))
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr(img: Image.Image) -> str:
    img = preprocess(img)
    return pytesseract.image_to_string(
        img,
        lang="eng",
        config="--psm 6"
    )

def split_lr(img: Image.Image):
    w, h = img.size
    return img.crop((0, 0, w//2, h)), img.crop((w//2, 0, w, h))

# --------------------------------------------------
# PARSING (ROBUST, TOLERANT)
# --------------------------------------------------
def find_pct(pattern, text):
    vals = re.findall(pattern, text, re.IGNORECASE)
    out = []
    for v in vals:
        try:
            out.append(float(v.replace(",", "")))
        except:
            pass
    return out

def best(vals, mode="max"):
    if not vals:
        return None
    if mode == "min":
        return min(vals)
    return max(vals)

def extract_stats(text: str) -> dict:
    s = {}

    s["tough"] = best(find_pct(r"Tough\s*\+?([\d,]+)%", text))
    s["puncture"] = best(find_pct(r"Puncture\s*\+?([\d,]+)%", text))
    s["crit_rate"] = best(find_pct(r"Crit\s*Rate\s*\+?([\d,]+)%", text))
    s["crit_damage"] = best(find_pct(r"Crit\s*Damage\s*\+?([\d,]+)%", text))

    for u, name in [("inf","Infantry"),("cav","Cavalry"),("arc","Archer")]:
        s[f"{u}_atk"] = best(find_pct(fr"{name}.*Attack.*\+?([\d,]+)%", text))
        s[f"{u}_hp"]  = best(find_pct(fr"{name}.*Health.*\+?([\d,]+)%", text))
        s[f"{u}_recv"] = best(
            find_pct(fr"{name}.*Received.*(-?[\d,]+)%", text),
            mode="min"
        )

    return {k:v for k,v in s.items() if v is not None}

def merge(a, b):
    out = dict(a)
    for k,v in b.items():
        if k not in out:
            out[k] = v
        else:
            if "recv" in k:
                out[k] = min(out[k], v)
            else:
                out[k] = max(out[k], v)
    return out

# --------------------------------------------------
# POWER MODEL (STABLE)
# --------------------------------------------------
def unit_power(d, u):
    atk = d.get(f"{u}_atk", 0)
    hp  = d.get(f"{u}_hp", 0)
    recv = d.get(f"{u}_recv", -500)

    tough = d.get("tough", 0)
    punct = d.get("puncture", 0)
    critr = d.get("crit_rate", 0)
    critd = d.get("crit_damage", 0)

    offense = atk * (1 + punct/100) * (1 + critr/100 * critd/100)
    recv_factor = max(0.2, 1 - abs(recv)/1200)
    defense = hp * (1 + tough/100) * recv_factor

    return offense * np.sqrt(defense + 1)

def auto_mix(r):
    w = {k: max(v, 0.1)**2 for k,v in r.items()}
    s = sum(w.values())
    mix = {k: int(round(w[k]/s*100)) for k in w}
    diff = 100 - sum(mix.values())
    best = max(mix, key=lambda k: mix[k])
    mix[best] += diff
    return mix

# --------------------------------------------------
# UI
# --------------------------------------------------
with st.expander("📌 OCR Tips"):
    st.write(
        "- Nutze **Complete** (nicht Concise)\n"
        "- Zoom im Spiel\n"
        "- Kein Scroll-Blur\n"
        "- Mehrere Stats-Seiten hochladen"
    )

files = st.file_uploader(
    "📤 Upload battle screenshots",
    type=["png","jpg","jpeg"],
    accept_multiple_files=True
)

if st.button("🔍 Analyze"):
    if not files:
        st.error("Bitte Screenshots hochladen")
        st.stop()

    my = {}
    enemy = {}
    debug = {}

    with st.spinner("Running OCR…"):
        for f in files:
            img = Image.open(io.BytesIO(f.getvalue()))
            l, r = split_lr(img)

            tl = ocr(l)
            tr = ocr(r)

            my = merge(my, extract_stats(tl))
            enemy = merge(enemy, extract_stats(tr))

            debug[f.name] = {"left": tl[:400], "right": tr[:400]}

    if not my or not enemy:
        st.error("OCR hat keine brauchbaren Werte gefunden.")
        st.stop()

    ratio = {}
    for u in ["inf","cav","arc"]:
        ratio[u] = unit_power(my,u) / max(unit_power(enemy,u),1)

    mix = auto_mix(ratio)

    st.success("✅ Analyse abgeschlossen")

    st.subheader("📌 Empfohlener Truppen-Mix")
    st.write(f"🐎 Cavalry: **{mix['cav']}%**")
    st.write(f"🏹 Archer: **{mix['arc']}%**")
    st.write(f"🛡 Infantry: **{mix['inf']}%**")

    st.subheader("📊 Vorteil")
    st.json({k: round(v,3) for k,v in ratio.items()})

    with st.expander("🔍 OCR Debug"):
        for k,v in debug.items():
            st.markdown(f"### {k}")
            st.text(v["left"])
            st.text(v["right"])

    st.download_button(
        "⬇️ Download JSON",
        json.dumps({"me":my,"enemy":enemy},indent=2),
        "battle_data.json",
        "application/json"
    )
