import os
import random
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from docx import Document
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_glyphs(image_path):
    """Extracts transparent handwriting glyphs from an uploaded image."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glyphs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 8 and h > 12:
            crop = img[y:y+h, x:x+w]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
            
            # Make background transparent
            datas = crop_pil.getdata()
            newData = []
            for item in datas:
                if item[0] > 180 and item[1] > 180 and item[2] > 180:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
            crop_pil.putdata(newData)
            glyphs.append(crop_pil)
    return glyphs


def generate_handwritten_page(docx_path, glyphs, output_path):
    """Parses DOCX and renders onto paper canvas with jitter."""
    doc = Document(docx_path)
    canvas_w, canvas_h = 1200, 1600
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (253, 252, 248, 255))

    margin_x = 100
    cursor_x = margin_x
    cursor_y = 120
    line_spacing = 60

    for p in doc.paragraphs:
        text = p.text
        if not text:
            cursor_y += line_spacing
            cursor_x = margin_x
            continue

        for char in text:
            if char == " ":
                cursor_x += random.randint(22, 32)
                if cursor_x > canvas_w - margin_x:
                    cursor_x = margin_x
                    cursor_y += line_spacing
                continue

            glyph = random.choice(glyphs).copy()

            # Randomize size, rotation, and baseline position
            scale = random.uniform(0.9, 1.1)
            glyph = glyph.resize((max(1, int(glyph.width * scale)), max(1, int(glyph.height * scale))), Image.Resampling.LANCZOS)
            glyph = glyph.rotate(random.uniform(-3.0, 3.0), expand=True, resample=Image.Resampling.BICUBIC)
            y_jitter = random.randint(-3, 3)

            if cursor_x + glyph.width > canvas_w - margin_x:
                cursor_x = margin_x
                cursor_y += line_spacing

            canvas.alpha_composite(glyph, (cursor_x, cursor_y + y_jitter))
            cursor_x += glyph.width + random.randint(1, 4)

        cursor_x = margin_x
        cursor_y += line_spacing

    final_img = Image.new("RGB", canvas.size, (255, 255, 255))
    final_img.paste(canvas, mask=canvas.split()[3])
    final_img.save(output_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        docx_file = request.files.get('docx_file')
        sample_files = request.files.getlist('sample_files')

        if not docx_file or not sample_files:
            return "Please upload both a Word document and at least one handwriting sample.", 400

        # Save uploaded DOCX
        docx_path = os.path.join(app.config['UPLOAD_FOLDER'], docx_file.filename)
        docx_file.save(docx_path)

        # Collect glyphs from all uploaded handwriting sample photos
        all_glyphs = []
        for file in sample_files:
            if file.filename != '':
                sample_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(sample_path)
                all_glyphs.extend(extract_glyphs(sample_path))

        if not all_glyphs:
            return "Could not extract handwriting from image. Try a clearer image with higher contrast.", 400

        # Generate handwritten output image
        output_filename = "result.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        generate_handwritten_page(docx_path, all_glyphs, output_path)

        return render_template('index.html', result_image=output_filename)

    return render_template('index.html', result_image=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
