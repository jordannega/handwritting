import random
import cv2
import numpy as np
from docx import Document
from PIL import Image, ImageOps

# ==========================================
# STAGE 1: GLYPH EXTRACTION VIA OPENCV
# ==========================================

def extract_glyphs_from_sample(image_path):
    """
    Loads a handwriting sample, isolates letter strokes, 
    and returns transparent RGBA glyph images.
    """
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding: invert so strokes are white, background is black
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Find contours around each written letter
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glyphs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out tiny noise artifacts
        if w > 8 and h > 12:
            # Crop stroke bounding box
            crop = img[y:y+h, x:x+w]
            
            # Convert crop to RGBA and make background transparent
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
            datas = crop_pil.getdata()
            
            newData = []
            for item in datas:
                # Turn light pixels (background) fully transparent
                if item[0] > 180 and item[1] > 180 and item[2] > 180:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            
            crop_pil.putdata(newData)
            glyphs.append(crop_pil)
            
    return glyphs


# ==========================================
# STAGE 2: PROCEDURAL RENDERING ENGINE
# ==========================================

def docx_to_handwriting(docx_path, output_image_path, glyph_pool):
    """
    Parses a DOCX file and pastes glyphs onto a lined canvas 
    with human-like jitter effect.
    """
    doc = Document(docx_path)
    
    # Create white canvas (Letter/A4 proportional dimensions)
    canvas_w, canvas_h = 1200, 1600
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (250, 250, 245, 255))

    # Margins and line tracking
    margin_x = 100
    cursor_x = margin_x
    cursor_y = 120
    line_spacing = 60

    if not glyph_pool:
        print("[!] No glyphs extracted. Check sample image thresholding.")
        return

    for p in doc.paragraphs:
        text = p.text
        if not text:
            cursor_y += line_spacing  # Empty paragraph / new line
            cursor_x = margin_x
            continue

        for char in text:
            if char == " ":
                cursor_x += random.randint(20, 30)  # Variable word spacing
                if cursor_x > canvas_w - margin_x:
                    cursor_x = margin_x
                    cursor_y += line_spacing
                continue

            # Pick a glyph variation from pool
            glyph = random.choice(glyph_pool).copy()

            # --- Jitter Simulation ---
            # 1. Size jitter (+/- 10%)
            scale_factor = random.uniform(0.9, 1.1)
            new_w = max(1, int(glyph.width * scale_factor))
            new_h = max(1, int(glyph.height * scale_factor))
            glyph = glyph.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 2. Rotation jitter (+/- 3 degrees)
            rot_angle = random.uniform(-3.0, 3.0)
            glyph = glyph.rotate(rot_angle, expand=True, resample=Image.Resampling.BICUBIC)

            # 3. Baseline shift (+/- 2 pixels)
            y_jitter = random.randint(-2, 2)

            # Line wrapping
            if cursor_x + glyph.width > canvas_w - margin_x:
                cursor_x = margin_x
                cursor_y += line_spacing

            # Paste glyph onto canvas
            canvas.alpha_composite(glyph, (cursor_x, cursor_y + y_jitter))

            # Advance X position with slight variable kerning
            cursor_x += glyph.width + random.randint(1, 4)

        # Move to next line after finishing paragraph
        cursor_x = margin_x
        cursor_y += line_spacing

    # Convert canvas back to RGB and save final image
    final_output = Image.new("RGB", canvas.size, (255, 255, 255))
    final_output.paste(canvas, mask=canvas.split()[3])
    final_output.save(output_image_path)
    print(f"[+] Render complete! Saved output to: {output_image_path}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Extract glyphs from your sample page
    extracted_glyphs = extract_glyphs_from_sample("sample_handwriting.png")
    
    # 2. Render input DOCX to hand-written page
    docx_to_handwriting("document.docx", "handwritten_output.png", extracted_glyphs)
