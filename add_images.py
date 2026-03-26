"""
add_images.py
-------------
Two ways to add missing product images:

METHOD A — Paste direct image URLs into the MANUAL_URLS dict below,
           then run:  python add_images.py

METHOD B — Save image files manually into static/images/ with the
           exact filename shown in the MISSING PRODUCTS list printed
           when you run this script, then run:  python add_images.py

The script will download all URLs in METHOD A and update products.csv.
"""

import csv, os, re, requests, time

CSV_PATH = "data/products.csv"          # must be in same folder as this script
IMG_DIR  = "static/images"
HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ══════════════════════════════════════════════════════════════════════════════
# METHOD A — Paste image URLs here.
# Key   = exact product_name from products.csv
# Value = direct image URL (right-click image → Copy image address)
#
# HOW TO GET THE URL:
#   1. Google the product name
#   2. Open the product page (Amazon / Nykaa / brand site)
#   3. Right-click the main product photo → "Copy image address"
#   4. Paste as the value below
# ══════════════════════════════════════════════════════════════════════════════
MANUAL_URLS = {
    # ── Moisturizers ──────────────────────────────────────────────────────────
    "CeraVe Moisturizing Cream":           "https://www.ceraveindia.com/-/media/project/loreal/brand-sites/cerave/americas/in/scx/products/pdp/packshots/moisturising-cream/moisturising-cream-50ml-lg.jpg?rev=-1?w=500&hash=E6DDC934ABF97E33F2F955D660ACB12C",
    "Pond's Light Moisturiser":            "https://ponds.in/cdn/shop/products/Fop_9749b0a5-e112-4044-ac12-6e458ef016b0.jpg?v=1646636462&width=1000",
    "Minimalist Sepicalm Moisturizer":     "https://sfycdn.speedsize.com/56385b25-4e17-4a9a-9bec-c421c18686fb/beminimalist.co/cdn/shop/files/Sepicalm_New.png?crop=center&height=1260&v=1721398128&width=840",
    "Aveeno Daily Moisturizing Lotion":    "https://images.ctfassets.net/aub2fvcyp2t8/5g9WdT0zRAK9FS8M6nbX6K/54333cbf2f5787d78d8295c3bfa1286e/dm_lotion-_354ml_front-en-in?fm=webp&w=1024",
    "Lakme Peach Milk Moisturizer":        "https://www.lakmeindia.com/cdn/shop/files/12893_H-8901030974328_1000x.jpg?v=1696485001",
    "Simple Hydrating Light Moisturiser":  "https://www.simpleskincare.in/cdn/shop/files/Module00_1000x1000.jpg?v=1769511700",
    "Dot & Key 72hr Hydrating Gel":        "https://www.dotandkey.com/cdn/shop/files/1-72-hrs-gel---listing.jpg?v=1744369744&width=700",
    "Vaseline Deep Moisture Cream":        "https://images-static.nykaa.com/media/catalog/product/5/1/510488e8901030769443_1.jpg?tr=w-344,h-344,cm-pad_resize",
    "L'Oreal Hydra Genius Aloe Water":     "https://m.media-amazon.com/images/I/61LEwHOYyUL._SY879_.jpg",
    "Re'equil Ceramide Moisturizer":       "https://www.reequil.com/cdn/shop/files/Ceramide_Hyaluronic_Acid_Moisturiser.png?v=1770667564&width=1920",

    # ── Face Washes ───────────────────────────────────────────────────────────
    "The Derma Co 2% Sali-Cinamide Face Wash":      "https://thedermaco.com/cdn/shop/files/1_pdp_2_sali_cinamide_new.jpg?v=1767606411&width=713",
    "Himalaya Oil Clear Lemon Face Wash":            "https://himalayawellness.in/cdn/shop/files/7555390-4_Himalaya-Oil-Clear-Lemon-Face-Wash-50-ml-Tube-3D_FOP_1800x1800.jpg?v=1737613147",
    "Chemist At Play Oil & Acne Control Face Wash":  "https://m.media-amazon.com/images/I/51dCh5yZFHL._SX522_.jpg",
    "Minimalist Salicylic Acid Cleanser":            "https://sfycdn.speedsize.com/56385b25-4e17-4a9a-9bec-c421c18686fb/beminimalist.co/cdn/shop/files/SalicylicCleanserNew.jpg?crop=center&height=1260&v=1756796206&width=840",
    "Cetaphil Oily Skin Cleanser":                   "https://www.cetaphil.in/dw/image/v2/BGGN_PRD/on/demandware.static/-/Sites-galderma-in-m-catalog/default/dw2803e62d/OSC%20Revive%20A+/OSC%20236ml/ATF/1.%20FoP.png?sw=900&sh=900&sm=fit&q=85",
    "CeraVe Hydrating Cleanser":                     "https://www.ceraveindia.com/-/media/project/loreal/brand-sites/cerave/americas/in/scx/products/pdp/packshots/hydrating-cleanser/hydrating-cleanser-473ml-lg.jpg?rev=-1?w=500&hash=399E1E55F5010A0E7C2400B61E0DB721",
    "Himalaya Aloe Vera Face Wash":                  "https://dailyglowmart.com/wp-content/uploads/2025/04/7405112-2_Moisturising-Aloe-Vera-Face-Wash_100ml_FOP_1800x1800.webp",
    "Hyphen Moisturizing Creamy Cleanser":           "https://m.media-amazon.com/images/I/41eKh6SpoML._SX522_.jpg",
    "NIVEA Milk Delights Face Wash":                 "https://img.nivea.com/-/media/miscellaneous/media-center-items/5/6/7/df5a1ab0f3484a37a717a387f1f8e9db-web_1010x1180_transparent_png.webp?mw=960&hash=DCA99D7C616939D6F5A2420D1B462A44",
    "Aquahance Moisture Surge Face Wash":            "https://images.apollo247.in/pub/media/catalog/product/A/Q/AQU0436_1.jpg?tr=w-264,q-80,f-webp,dpr-false,c-at_max",
    "Lakme Blush & Glow Face Wash":                  "https://www.lakmeindia.com/cdn/shop/files/12958_S1-8901030994739_1000x.jpg?v=1720091661",
    "Bella Vita Organic Face Wash":                  "https://www.distacart.com/cdn/shop/files/1_Pack_of_2-Photoroom_600x.png?v=1754652952",
    "Garnier Vitamin C Face Wash":                   "https://www.garnier.in/-/media/project/loreal/brand-sites/garnier/apac/in/products/light-complete/light-complete-facewash/light-complete-facewash-150g/8901526593859-n1.jpg?w=500&rev=1f289f5fb64b4d3b86abeed8c4fdc2d3&hash=CF435A87A4C60D4A3AA1B237AD5F0223",
    "Plum Simply Bright Face Wash":                  "https://m.media-amazon.com/images/I/41UOZOZKBOL._SX522_.jpg",
    "Simple Refreshing Facial Wash":                 "https://m.media-amazon.com/images/I/51wqZYWGr+L._SX522_.jpg",

    # ── Sunscreens ────────────────────────────────────────────────────────────
    "CeraVe Invisible Mineral Sunscreen SPF 50, Face Sunscreen for Sensitive Skin With Zinc Oxide & Titanium Dioxide, Vitamin E + Niacinamide + Ceramides, Oil Free, Travel Size 1.62 oz": "https://www.cerave.com/-/media/project/loreal/brand-sites/cerave/americas/us/sunscreen/face/invisible-sunscreen-pdp/700x785/invisible-mineral-sunscreen3-700x785-v1.jpg?rev=e57ad633b257498abbb7c146aeeafd96&w=900&hash=74ED971D5269719C96B9ABE68EE82FE7",
    "Innisfree Daily UV Defense SPF 36  Korean Face Sunscreen": "https://m.media-amazon.com/images/I/41irOnF3T2L._SX342_SY445_QL70_FMwebp_.jpg",
    "EltaMD UV Clear Face Sunscreen SPF 46":         "https://eltamd.com/cdn/shop/files/UV_Clear_SPF_46_02500A_With_award_650x.jpg?v=1773888112",
    "Fixderma Shadow Sunscreen SPF 50+ PA+++ Cream | Sunscreen for Dry Skin": "https://www.fixderma.com/cdn/shop/files/FDSHADOW50PLUSCREAM75ML30088.webp?v=1756361946",
    "Neutrogena Ultra Sheer Sunscreen SPF 50+":      "https://images.ctfassets.net/aub2fvcyp2t8/4oa6YGDxctTkbVmX3sVA9i/cfa7d387860a2ef2c96d4a0a982f1e59/neutrogena-ultra-sheer-dry-touch-sunblock-spf-50-front1-en-in?fm=webp&w=1024",
    "The Derma Co 1% Hyaluronic Sunscreen Aqua Gel SPF 50 PA++++": "https://thedermaco.com/cdn/shop/files/1_PDP_50g_new.jpg?v=1770277899&width=713",
    "Hydrating Sunscreen for Dry Skin (50gm) Lightweight, Photostable Sunscreen SPF 50": "https://m.media-amazon.com/images/I/51+H1QtcK2L._SX522_.jpg",
    "Aqualogica Glow+ Dewy Gel Sunscreen - 50 g":   "https://aqualogica.in/cdn/shop/files/BOPwhiteBG_59907c76-b796-4e25-9a03-7dcb8a6eceee.jpg?v=1773220798&width=600",
    "Neutrogena Ultra sheer Sunscreen, SPF 50+":     "https://images.ctfassets.net/aub2fvcyp2t8/4oa6YGDxctTkbVmX3sVA9i/cfa7d387860a2ef2c96d4a0a982f1e59/neutrogena-ultra-sheer-dry-touch-sunblock-spf-50-front1-en-in?fm=webp&w=1024",
    "Cetaphil Sun SPF 50+ Light Gel":                "https://m.media-amazon.com/images/I/61BrU7Gym-L._SX522_.jpg",
    "Dot & Key Mango Detan Gel Sunscreen SPF 50":    "https://m.media-amazon.com/images/I/61-QPC7fzvL._SX522_.jpg",
}

# ─────────────────────────────────────────────────────────────────────────────

def slugify(name):
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_-]+", "_", name)
    return name[:60]

def download(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"    ✅ Downloaded ({os.path.getsize(path)//1024} KB)")
        return True
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        return False

os.makedirs(IMG_DIR, exist_ok=True)

# Load CSV
with open(CSV_PATH, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())
if "image_local" not in fieldnames:
    fieldnames.append("image_local")

# Build lookup
row_by_name = {r["product_name"]: r for r in rows}

print("=" * 60)
print("MISSING PRODUCTS (need images):")
print("=" * 60)
missing = []
for r in rows:
    local = r.get("image_local", "")
    if not local or "fallback" in local:
        slug = slugify(r["product_name"])
        fname = f"{slug}.jpg"
        missing.append((r["product_name"], fname))
        print(f"  • {r['product_name']}")
        print(f"    → filename if dropping manually: {fname}")

print(f"\nTotal missing: {len(missing)}")
print("=" * 60)

# Process MANUAL_URLS
updated = 0
for name, img_url in MANUAL_URLS.items():
    if not img_url.strip():
        continue
    if name not in row_by_name:
        print(f"\n⚠  Name not found in CSV: '{name}'")
        continue

    row   = row_by_name[name]
    slug  = slugify(name)
    fpath = os.path.join(IMG_DIR, slug + ".jpg")

    print(f"\n→ {name}")
    if download(img_url, fpath):
        row["image_local"] = f"/static/images/{slug}.jpg"
        updated += 1
    time.sleep(0.5)

# Also auto-detect manually dropped files
for name, fname in missing:
    fpath = os.path.join(IMG_DIR, fname)
    if os.path.exists(fpath):
        row_by_name[name]["image_local"] = f"/static/images/{fname}"
        print(f"✅ Auto-detected manual file: {fname}")
        updated += 1

# Save CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

still_missing = sum(1 for r in rows if not r.get("image_local") or "fallback" in r.get("image_local",""))
print(f"\n✅ Updated {updated} products.")
print(f"📋 Still missing: {still_missing}/45")
print(f"💾 CSV saved: {CSV_PATH}")