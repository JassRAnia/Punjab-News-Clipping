import os
import json
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
# Date formatting for today
today_str = datetime.now().strftime("%d-%m-%Y")
# E.g. "22-04-2026"
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
    client = genai.Client() # Uses GEMINI_API_KEY env var automatically
    
    print("Uploading images to Gemini...")
    uploaded_files = []
    for path in image_paths:
        f = client.files.upload(file=path)
        uploaded_files.append(f)
    
    print("Prompting Gemini for Analysis...")
    
    prompt = """
    You are an expert news analyst. Review the provided images of a daily Punjabi newspaper.
    Your task is to comprehensively analyze the pages and extract ONLY news articles that are related to the Punjab Government.
    
    For each relevant article, extract:
    1. Title (Translate to clear English)
    2. Summary (Provide a 3-5 sentence English summary of the news)
    3. Sentiment (Determine if the news is POSITIVE or NEGATIVE towards the Punjab Government)
    4. Page Number (Which page the clipping came from, just a rough indicator from the images given)
    
    Rules:
    - If a news item has nothing to do with the Punjab Government (e.g. general sports, foreign news, pure advertising), IGNORE IT.
    - Provide the final output strictly as a JSON list of objects without markdown formatting.
    - Here is the JSON structure:
    [
      {
        "title": "...",
        "summary": "...",
        "sentiment": "Positive" | "Negative",
        "page_number": "1"
      }
    ]
    """
    
    # We specify the JSON response schema for reliability
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[prompt] + uploaded_files,
        config=types.GenerateContentConfig(
            response_mimetype="application/json",
            temperature=0.2,
        ),
    )
    
    try:
        data = json.loads(response.text)
        return data
    except Exception as e:
        print(f"JSON Parsing error: {e}")
        print("Raw response:", response.text)
        # Attempt to clean code blocks if Gemini returns markdown JSON
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)

# ----------------- PDF GENERATION -----------------
def generate_pdf(filename, title, news_items):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
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
    
    # Title
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

    # 1. Capture e-paper images
    screenshot_paths = await capture_epaper_pages()
    
    if not screenshot_paths:
        print("No screenshots were captured. Cannot proceed.")
        return

    # 2. Extract news via Gemini
    news_items = analyze_images_with_gemini(screenshot_paths)
    print(f"Found {len(news_items)} relevant news items.")
    
    # 3. Separate logic
    positive_news = [item for item in news_items if item.get('sentiment', '').lower() == 'positive']
    negative_news = [item for item in news_items if item.get('sentiment', '').lower() == 'negative']
    
    # 4. Generate the two separate PDF reports
    os.makedirs("reports", exist_ok=True)
    generate_pdf(f"reports/Positive_News_{today_str}.pdf", "Positive Punjab Gov News", positive_news)
    generate_pdf(f"reports/Negative_News_{today_str}.pdf", "Negative Punjab Gov News", negative_news)

if __name__ == "__main__":
    asyncio.run(main())
