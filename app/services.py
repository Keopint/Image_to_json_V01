import pytesseract
import cv2
import numpy as np
import base64
from collections import defaultdict
from difflib import SequenceMatcher

import os
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata' 


def preprocess_for_ocr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    limg = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened


def analyze_webpage_image(img: np.ndarray) -> dict:
    
    processed_img = preprocess_image(img)
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Обнаружение текстовых элементов
    text_elements = detect_text_elements(gray)
    
    # 2. Обнаружение кнопок
    button_elements = detect_buttons(processed_img)
    
    # 3. Обнаружение полей ввода
    input_elements = detect_input_fields(processed_img)
    
    # 4. Обнаружение изображений
    image_elements = detect_images(processed_img)
    
    all_elements = text_elements + button_elements + input_elements + image_elements
    filtered_elements = filter_overlapping_elements(all_elements)
    
    return {
        "elements": filtered_elements,
        "width": img.shape[1],
        "height": img.shape[0]
    }

def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    limg = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return denoised


def detect_text_elements(gray_img):
    try:
        custom_config = r"--oem 3 --psm 6 -l rus+eng --tessdata-dir /usr/share/tesseract-ocr/4.00/tessdata"
        processed = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        processed = cv2.medianBlur(processed, 3)

        data = pytesseract.image_to_string(
            processed,
            config=custom_config,
            output_type=pytesseract.Output.DICT,
            timeout=30
        )

        elements = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 60 and data['text'][i].strip():
                text = data['text'][i].strip()
                elements.append({
                    "type": "text",
                    "text": text,
                    "position": {
                        "x": data['left'][i],
                        "y": data['top'][i],
                        "width": data['width'][i],
                        "height": data['height'][i]
                    }
                })
        
        return elements
    
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return []


def merge_text_lines(elements):
    lines = defaultdict(list)
    
    for elem in elements:
        line_key = round(elem['position']['y'] / max(elem['position']['height'], 1))
        lines[line_key].append(elem)
    
    merged_elements = []
    for line in lines.values():
        if len(line) == 1:
            merged_elements.append(line[0])
            continue
            
        line.sort(key=lambda e: e['position']['x'])
        
        # Объединяем текст и координаты
        combined_text = ' '.join(e['text'] for e in line)
        x = line[0]['position']['x']
        y = line[0]['position']['y']
        width = line[-1]['position']['x'] + line[-1]['position']['width'] - x
        height = max(e['position']['height'] for e in line)
        
        merged_elements.append({
            "type": "text",
            "text": combined_text,
            "position": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            }
        })
    
    return merged_elements


def detect_buttons(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    buttons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        if 1.5 < aspect_ratio < 5.0:
            roi = img[max(0,y-5):min(img.shape[0],y+h+5), 
                     max(0,x-5):min(img.shape[1],x+w+5)]
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(
                gray_roi, 
                config='--psm 8 -l rus+eng',  
                timeout=10
            ).strip()
            
            if text:
                text = text.replace('\n', ' ').replace('  ', ' ')
                buttons.append({
                    "type": "button",
                    "text": text,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "image_embedding": None
                })
    
    return buttons


def detect_input_fields(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    input_fields = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            
            if 20 < w < 500 and 10 < h < 100:
                roi = gray[max(0, y-30):y, x:x+w]
                text = pytesseract.image_to_string(roi, config='--psm 6').strip()
                
                input_fields.append({
                    "type": "input",
                    "label": text if text else None,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    }
                })
    
    return input_fields

def detect_images(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    images = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: 
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 5.0:
            image_roi = img[y:y+h, x:x+w]
            _, buffer = cv2.imencode('.jpg', image_roi)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            images.append({
                "type": "image",
                "position": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                },
                "image_embedding": image_base64
            })
    
    return images

def filter_overlapping_elements(elements):
    elements_sorted = sorted(elements, key=lambda x: (
        0 if x['type'] == 'button' else 
        1 if x['type'] == 'input' else 
        2 if x['type'] == 'image' else 3
    ))
    
    filtered = []
    for elem in elements_sorted:
        overlap = False
        for existing in filtered:
            if elements_overlap(elem, existing):
                overlap = True
                break
        if not overlap:
            filtered.append(elem)
    
    return filtered

def elements_overlap(a, b):
    a_pos = a['position']
    b_pos = b['position']
    
    return (a_pos['x'] < b_pos['x'] + b_pos['width'] and
            a_pos['x'] + a_pos['width'] > b_pos['x'] and
            a_pos['y'] < b_pos['y'] + b_pos['height'] and
            a_pos['y'] + a_pos['height'] > b_pos['y'])
    
    
def compare_pages(img1: np.ndarray, img2: np.ndarray) -> dict:
    page1 = analyze_webpage_image(img1)
    page2 = analyze_webpage_image(img2)
    
    def create_index(elements):
        index = defaultdict(list)
        for elem in elements:
            if elem['type'] == 'text':
                index['text'].append(elem)
            else:
                index['other'].append(elem)
        return index
    
    index1 = create_index(page1['elements'])
    index2 = create_index(page2['elements'])
    
    differences = []
    
    for elem in page1['elements']:
        if not find_similar_element(elem, page2['elements']):
            differences.append({
                "change_type": "removed",
                "old_position": elem['position'],
                "new_position": None,
                "old_text": elem.get('text'),
                "new_text": None,
                "similarity_score": 0.0
            })
    
    for elem in page2['elements']:
        if not find_similar_element(elem, page1['elements']):
            differences.append({
                "change_type": "added",
                "old_position": None,
                "new_position": elem['position'],
                "old_text": None,
                "new_text": elem.get('text'),
                "similarity_score": 0.0
            })
    
    for elem1 in page1['elements']:
        elem2 = find_similar_element(elem1, page2['elements'])
        if elem2:
            similarity = compare_elements(elem1, elem2)
            if similarity < 0.9:  
                diff = {
                    "change_type": "modified",
                    "old_position": elem1['position'],
                    "new_position": elem2['position'],
                    "old_text": elem1.get('text'),
                    "new_text": elem2.get('text'),
                    "similarity_score": similarity
                }
                
                if (abs(elem1['position']['x'] - elem2['position']['x']) > 10 or 
                    abs(elem1['position']['y'] - elem2['position']['y']) > 10):
                    diff["change_type"] = "moved"
                
                differences.append(diff)
    
    stats = {
        "added_count": sum(1 for d in differences if d['change_type'] == 'added'),
        "removed_count": sum(1 for d in differences if d['change_type'] == 'removed'),
        "modified_count": sum(1 for d in differences if d['change_type'] == 'modified'),
        "moved_count": sum(1 for d in differences if d['change_type'] == 'moved')
    }
    
    return {
        "differences": differences,
        **stats
    }

def find_similar_element(element, elements_list, threshold=0.7):
    for candidate in elements_list:
        if element['type'] != candidate['type']:
            continue
            
        if element['type'] == 'text':
            similarity = SequenceMatcher(
                None, 
                element.get('text', ''), 
                candidate.get('text', '')
            ).ratio()
            
            if similarity > threshold:
                return candidate
        else:
            pos_similarity = position_similarity(element['position'], candidate['position'])
            if pos_similarity > threshold:
                return candidate
    return None

def compare_elements(elem1, elem2):
    if elem1['type'] == 'text' and elem2['type'] == 'text':
        text_sim = SequenceMatcher(
            None, 
            elem1.get('text', ''), 
            elem2.get('text', '')
        ).ratio()
        pos_sim = position_similarity(elem1['position'], elem2['position'])
        return (text_sim + pos_sim) / 2
    else:
        return position_similarity(elem1['position'], elem2['position'])

def position_similarity(pos1, pos2):
    x_diff = 1 - abs(pos1['x'] - pos2['x']) / max(pos1['width'], pos2['width'], 1)
    y_diff = 1 - abs(pos1['y'] - pos2['y']) / max(pos1['height'], pos2['height'], 1)
    w_diff = 1 - abs(pos1['width'] - pos2['width']) / max(pos1['width'], pos2['width'], 1)
    h_diff = 1 - abs(pos1['height'] - pos2['height']) / max(pos1['height'], pos2['height'], 1)
    return (x_diff + y_diff + w_diff + h_diff) / 4


def visualize_differences(img1, img2, differences):
    result = img2.copy()
    
    for diff in differences:
        if diff['change_type'] == 'added':
            color = (0, 255, 0)  
        elif diff['change_type'] == 'removed':
            color = (0, 0, 255)  
        else:
            color = (255, 255, 0)  
            
        pos = diff.get('new_position') or diff.get('old_position')
        if pos:
            cv2.rectangle(
                result,
                (pos['x'], pos['y']),
                (pos['x'] + pos['width'], pos['y'] + pos['height']),
                color,
                2
            )
            
            cv2.putText(
                result,
                diff['change_type'],
                (pos['x'], pos['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
    
    return result