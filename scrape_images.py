"""
scrape_images.py
----------------
Fetches a product image for every row in products.csv by scraping
the og:image / twitter:image meta tag from each product's URL.

Saves images to:  static/images/<slugified_product_name>.jpg
Updates CSV to:   products.csv  (adds/updates 'image_url' column)

Run once:  python scrape_images.py
"""

import csv
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ── Config ────────────────────────────────────────────────────────────────────
CSV_IN        = "data/products.csv"
CSV_OUT       = "data/products.csv"          # overwrites in place
IMG_DIR       = "static/images"         # folder to save images
DELAY_SECONDS = 1.5                     # polite delay between requests
TIMEOUT       = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Category fallback icons (emoji rendered as SVG by the frontend)
CATEGORY_FALLBACK = {
    "Moisturizer": "/static/fallback_moisturizer.png",
    "Face Wash":   "/static/fallback_facewash.png",
    "Sunscreen":   "/static/fallback_sunscreen.png",
}

os.makedirs(IMG_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    """Convert product name to a safe filename."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_-]+", "_", name)
    return name[:60]


def fetch_og_image(url: str) -> str | None:
    """Try to get og:image or twitter:image from a product page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Priority order of meta tags to check
        candidates = [
            soup.find("meta", property="og:image"),
            soup.find("meta", attrs={"name": "twitter:image"}),
            soup.find("meta", attrs={"name": "twitter:image:src"}),
            soup.find("meta", property="og:image:secure_url"),
        ]
        for tag in candidates:
            if tag and tag.get("content"):
                img_url = tag["content"].strip()
                # Make relative URLs absolute
                if img_url.startswith("//"):
                    img_url = "https:" + img_url
                elif img_url.startswith("/"):
                    img_url = urljoin(url, img_url)
                return img_url

        # Last resort: find first <img> with reasonable size hint
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src and any(ext in src for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                if img_url.startswith("//"):
                    src = "https:" + src
                elif src.startswith("/"):
                    src = urljoin(url, src)
                return src

    except Exception as e:
        print(f"    ⚠ Could not fetch page: {e}")
    return None


def download_image(img_url: str, save_path: str) -> bool:
    """Download image from URL and save locally. Returns True on success."""
    try:
        r = requests.get(img_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "image" not in content_type and "octet" not in content_type:
            print(f"    ⚠ Not an image (Content-Type: {content_type})")
            return False
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        size_kb = os.path.getsize(save_path) / 1024
        print(f"    ✅ Saved ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"    ⚠ Download failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(CSV_IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    if "image_local" not in fieldnames:
        fieldnames.append("image_local")

    total = len(rows)
    success = 0

    for i, row in enumerate(rows, 1):
        name = row["product_name"]
        url  = row["url"]
        cat  = row["category"]
        slug = slugify(name)

        print(f"\n[{i}/{total}] {name}")
        print(f"  URL: {url[:70]}...")

        # Check if already downloaded
        existing = row.get("image_local", "")
        if existing and os.path.exists(existing.lstrip("/")):
            print(f"  ✔ Already have image, skipping")
            success += 1
            continue

        # Determine save path (try jpg first, fallback to png)
        save_path = os.path.join(IMG_DIR, slug + ".jpg")

        # Step 1: get og:image URL from product page
        print(f"  → Fetching product page...")
        img_url = fetch_og_image(url)

        if img_url:
            print(f"  → Image URL: {img_url[:70]}...")
            # Step 2: download the image
            ok = download_image(img_url, save_path)
            if ok:
                row["image_local"] = f"/static/images/{slug}.jpg"
                success += 1
            else:
                row["image_local"] = CATEGORY_FALLBACK.get(cat, "")
        else:
            print(f"  ✗ No image found — using category fallback")
            row["image_local"] = CATEGORY_FALLBACK.get(cat, "")

        time.sleep(DELAY_SECONDS)

    # Write updated CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*50}")
    print(f"Done! {success}/{total} images fetched successfully.")
    print(f"CSV updated: {CSV_OUT}")
    print(f"Images saved to: {IMG_DIR}/")


if __name__ == "__main__":
    main()