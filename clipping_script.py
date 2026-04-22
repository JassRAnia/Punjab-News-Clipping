import os
import json
import base64
import asyncio
from datetime import datetime
from playwright.async_api import async_playwright
from google import genai
from google.genai import types
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor

# ----------------- CONFIGURATION -----------------
today_str = datetime.now().strftime("%d-%m-%Y")
BASE_URL = f"https://www.rozanaspokesman.com/epaper/{today_str}"
TOTAL_PAGES = 12
SCREENSHOT_DIR = "screenshots"

# ----------------- PLAYWRIGHT SCRAPING -----------------
async def capture_epaper_pages():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    screenshot_paths = []

    print(f"Scraping e-paper for {today_str}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 1080})
        page = await context.new_page()

        for i in range(1, TOTAL_PAGES + 1):
            url = f"{BASE_URL}/{i}/punjab"
            print(f"Loading {url}")
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
                path = os.path.join(SCREENSHOT_DIR, f"page_{i}.png")
                await page.screenshot(path=path, full_page=True)
                screenshot_paths.append(path)
                print(f"Saved {path}")
            except Exception as e:
                print(f"Error saving page {i}: {e}")

        await browser.close()
    return screenshot_paths

# ----------------- GEMINI AI CLASSIFICATION -----------------
def analyze_images_with_gemini(image_paths):
    # gemini-2.0-flash: confirmed available in google-genai SDK with free tier
    # Images sent inline as base64 in batches of 4 to stay within token limits
    client = genai.Client()

    prompt = """
You are an expert news analyst. Review the provided newspaper page images (from the Rozana Spokesman Punjabi daily).
Extract ONLY news articles related to the Punjab Government (AAP government of Punjab, Punjab CM, Punjab ministers, Punjab state policies, Punjab departments etc.).

For each relevant article output:
1. title     - Translated to clear English
2. summary   - 3-5 sentence English summary
3. sentiment - "Positive" or "Negative" towards the Punjab Government
4. page_number - approximate page number (use image order: 1st image = page listed in filename)

Rules:
- IGNORE sports, foreign news, advertisements, national politics unrelated to Punjab govt.
- Return ONLY a valid JSON array. No markdown, no code fences, no explanation.
- If nothing is relevant, return: []

Example:
[
  {
    "title": "Punjab Govt Launches Free Bus Service for Women",
    "summary": "The AAP government in Punjab announced...",
    "sentiment": "Positive",
    "page_number": "2"
  }
]
"""

    all_news = []
    BATCH_SIZE = 4  # 4 pages per API call to stay within free-tier token limits

    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[batch_start:batch_start + BATCH_SIZE]
        print(f"Analyzing pages {batch_start + 1} to {batch_start + len(batch)}...")

        # Build inline image parts from base64
        contents = [prompt]
        for idx, path in enumerate(batch):
            with open(path, "rb") as f:
                raw_bytes = f.read()
            contents.append(
                types.Part.from_bytes(
                    data=raw_bytes,
                    mime_type="image/png"
                )
            )
            contents.append(f"[Above image is page {batch_start + idx + 1}]")

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            raw = response.text.strip()
            # Strip markdown code fences if model ignores mime_type hint
            if raw.startswith("```"):
                raw = raw[raw.index("["):]
                raw = raw[:raw.rindex("]") + 1]
            batch_data = json.loads(raw)
            if isinstance(batch_data, list):
                all_news.extend(batch_data)
                print(f"  -> Found {len(batch_data)} relevant items in this batch.")
        except Exception as e:
            print(f"Error analyzing batch (pages {batch_start+1}-{batch_start+len(batch)}): {e}")

    return all_news

# ----------------- PDF GENERATION -----------------
def generate_pdf(filename, title, news_items):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=14,
        textColor=HexColor('#2c3e50')
    )

    item_title_style = ParagraphStyle(
        'ItemTitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6,
        textColor=HexColor('#2980b9')
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        leading=16
    )

    content = []
    content.append(Paragraph(f"{title} - {today_str}", title_style))
    content.append(Spacer(1, 12))

    if not news_items:
        content.append(Paragraph("No news items found for this category today.", body_style))
    else:
        for idx, item in enumerate(news_items, 1):
            title_text = f"{idx}. {item.get('title', 'Unknown Title')} (Page {item.get('page_number', '?')})"
            content.append(Paragraph(title_text, item_title_style))
            summary_text = item.get('summary', '')
            content.append(Paragraph(summary_text, body_style))
            content.append(Spacer(1, 10))

    try:
        doc.build(content)
        print(f"Generated {filename} successfully.")
    except Exception as e:
        print(f"Could not generate PDF: {e}")

# ----------------- MAIN FLOW -----------------
async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set. Exiting.")
        return

    # 1. Capture e-paper screenshots
    screenshot_paths = await capture_epaper_pages()

    if not screenshot_paths:
        print("No screenshots were captured. Cannot proceed.")
        return

    # 2. Analyze with Gemini in batches
    print("Prompting Gemini for Analysis...")
    news_items = analyze_images_with_gemini(screenshot_paths)
    print(f"Found {len(news_items)} relevant news items in total.")

    # 3. Separate by sentiment
    positive_news = [item for item in news_items if item.get('sentiment', '').lower() == 'positive']
    negative_news = [item for item in news_items if item.get('sentiment', '').lower() == 'negative']

    print(f"  Positive: {len(positive_news)} | Negative: {len(negative_news)}")

    # 4. Generate PDF reports
    os.makedirs("reports", exist_ok=True)
    generate_pdf(f"reports/Positive_News_{today_str}.pdf", "Positive Punjab Gov News", positive_news)
    generate_pdf(f"reports/Negative_News_{today_str}.pdf", "Negative Punjab Gov News", negative_news)

if __name__ == "__main__":
    asyncio.run(main())
