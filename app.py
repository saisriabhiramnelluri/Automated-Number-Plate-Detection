import os
import ast
import csv
import cv2
import torch
import string
import numpy as np
import pandas as pd
import subprocess
from flask import Flask, render_template, request, Response, redirect, url_for
from ultralytics import YOLO
from sort.sort import Sort
import easyocr
from scipy.interpolate import interp1d

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
reader = easyocr.Reader(['en'], gpu=True)
processing_done = False


# ───────────────────────────────────────────────────────── #
# ------------------------ HELPERS ------------------------ #
# ───────────────────────────────────────────────────────── #

def license_complies_format(text):
    if len(text) != 7: return False
    valid = lambda ch, idx: (ch.isalpha() or ch in dict_int_to_char) if idx in [0, 1, 4, 5, 6] else (ch.isdigit() or ch in dict_char_to_int)
    return all(valid(text[i], i) for i in range(7))

dict_char_to_int = {'O': '0','I': '1','J': '3','A': '4','G': '6','S': '5'}
dict_int_to_char = {'0': 'O','1': 'I','3': 'J','4': 'A','6': 'G','5': 'S'}

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(7):
        if text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(crop):
    detections = reader.readtext(crop)
    for _, text, score in detections:
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, track_ids):
    x1, y1, x2, y2, *_ = license_plate
    for xcar1, ycar1, xcar2, ycar2, car_id in track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    return img

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                car_data = results[frame_nmr][car_id]
                if 'car' in car_data and 'license_plate' in car_data:
                    plate = car_data['license_plate']
                    if 'text' in plate:
                        f.write('{},{},{},{},{},{},{}\n'.format(
                            frame_nmr,
                            car_id,
                            '[{} {} {} {}]'.format(*car_data['car']['bbox']),
                            '[{} {} {} {}]'.format(*plate['bbox']),
                            plate['bbox_score'],
                            plate['text'],
                            plate['text_score']
                        ))

def interpolate_data(input_csv, output_csv):
    with open(input_csv, 'r') as file:
        data = list(csv.DictReader(file))

    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated = []
    for car_id in np.unique(car_ids):
        car_mask = car_ids == car_id
        frames = frame_numbers[car_mask]
        car_bboxes_ = []
        lp_bboxes_ = []

        first_frame = frames[0]
        for i in range(len(frames)):
            f = frames[i]
            car = car_bboxes[car_mask][i]
            lp = license_plate_bboxes[car_mask][i]
            if i > 0 and f - frames[i-1] > 1:
                gap = f - frames[i-1]
                x = np.array([frames[i-1], f])
                x_new = np.linspace(frames[i-1], f, gap, endpoint=False)
                car_interp = interp1d(x, np.vstack((car_bboxes_[-1], car)), axis=0)(x_new)
                lp_interp = interp1d(x, np.vstack((lp_bboxes_[-1], lp)), axis=0)(x_new)
                car_bboxes_.extend(car_interp[1:])
                lp_bboxes_.extend(lp_interp[1:])
            car_bboxes_.append(car)
            lp_bboxes_.append(lp)

        for i, bbox in enumerate(car_bboxes_):
            frame = first_frame + i
            row = {'frame_nmr': str(frame), 'car_id': str(car_id),
                   'car_bbox': ' '.join(map(str, bbox)),
                   'license_plate_bbox': ' '.join(map(str, lp_bboxes_[i]))}
            match = next((r for r in data if int(r['frame_nmr']) == frame and int(float(r['car_id'])) == car_id), None)
            if match:
                row['license_plate_bbox_score'] = match.get('license_plate_bbox_score', '0')
                row['license_number'] = match.get('license_number', '0')
                row['license_number_score'] = match.get('license_number_score', '0')
            else:
                row.update({'license_plate_bbox_score': '0', 'license_number': '0', 'license_number_score': '0'})
            interpolated.append(row)

    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score',
              'license_number', 'license_number_score']
    with open(output_csv, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=header).writeheader()
        csv.DictWriter(f, fieldnames=header).writerows(interpolated)

# ───────────────────────────────────────────────────────── #
# ------------------------ ROUTES ------------------------- #
# ───────────────────────────────────────────────────────── #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global processing_done
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)

    save_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
    file.save(save_path)

    # Skip processing if already done
    if processing_done:
        return redirect(url_for('output_page'))

    # Run detection pipeline
    coco = YOLO('yolov8n.pt')  # REMOVE .to(DEVICE)
    plate_model = YOLO('license_plate_detector.pt')  # REMOVE .to(DEVICE)
    tracker = Sort()
    results = {}
    cap = cv2.VideoCapture(save_path)
    frame_nmr = -1
    vehicles = [2, 3, 5, 7]

    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break
        results[frame_nmr] = {}
        detections = coco(frame)[0]  # Error was here due to bad .to()
        dets = [[*d[:4], d[4]] for d in detections.boxes.data.tolist() if int(d[5]) in vehicles]
        ids = tracker.update(np.asarray(dets))
        plates = plate_model(frame)[0]

        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, ids)
            if car_id != -1:
                crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, crop_thresh = cv2.threshold(crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                text, text_score = read_license_plate(crop_thresh)
                if text:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'bbox_score': score,
                            'text': text,
                            'text_score': text_score
                        }
                    }

    write_csv(results, 'test.csv')
    interpolate_data('test.csv', 'test_interpolated.csv')
        # ─── LOG FINAL PLATES TO visualized_log.csv ─── #
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    df = pd.read_csv('test_interpolated.csv')
    with open('visualized_log.csv', 'w') as f:
        f.write('frame_nmr,car_id,license_number,license_number_score\n')

    for car_id in np.unique(df['car_id']):
        max_score = np.amax(df[df['car_id'] == car_id]['license_number_score'])
        best_row = df[(df['car_id'] == car_id) & (df['license_number_score'] == max_score)]
        license_number = best_row['license_number'].iloc[0]
        best_frame = int(best_row['frame_nmr'].iloc[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
        ret, _ = cap.read()

        with open('visualized_log.csv', 'a') as log_file:
            log_file.write(f"{best_frame},{car_id},{license_number},{max_score:.4f}\n")

    processing_done = True
    return redirect(url_for('output_page'))

@app.route('/output')
def output_page():
    plates = []
    if os.path.exists('visualized_log.csv'):
        df = pd.read_csv('visualized_log.csv')
        plates = df['license_number'].dropna().unique().tolist()
    return render_template('output_ready.html', plates=plates)


@app.route('/video_feed')
def video_feed():
    return Response(generate_visualization(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_visualization():
    import ast
    df = pd.read_csv('test_interpolated.csv')
    cap = cv2.VideoCapture(os.path.join(UPLOAD_FOLDER, 'input.mp4'))

    license_plate = {}
    for car_id in np.unique(df['car_id']):
        max_ = np.amax(df[df['car_id'] == car_id]['license_number_score'])
        best_row = df[(df['car_id'] == car_id) & (df['license_number_score'] == max_)]
        license_plate_number = best_row['license_number'].iloc[0]
        best_frame = int(best_row['frame_nmr'].iloc[0])

        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
        ret, frame = cap.read()
        x1, y1, x2, y2 = ast.literal_eval(
            best_row['license_plate_bbox'].iloc[0]
            .replace('[ ', '[').replace('   ', ' ')
            .replace('  ', ' ').replace(' ', ',')
        )
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id] = {
            'license_crop': license_crop,
            'license_plate_number': license_plate_number
        }

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        df_ = df[df['frame_nmr'] == frame_nmr]

        for _, row in df_.iterrows():
            car_id = row['car_id']

            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                row['car_bbox'].replace('[ ', '[').replace('   ', ' ')
                .replace('  ', ' ').replace(' ', ',')
            )
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)),
                        (0, 255, 0), 25, line_length_x=200, line_length_y=200)

            x1, y1, x2, y2 = ast.literal_eval(
                row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ')
                .replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            try:
                license_crop = license_plate[car_id]['license_crop']
                plate_text = license_plate[car_id]['license_plate_number']
                H, W, _ = license_crop.shape

                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    plate_text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17
                )
                cv2.putText(frame,
                            plate_text,
                            (int((car_x2 + car_x1 - text_width) / 2),
                             int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)
            except:
                pass

        display_frame = cv2.resize(frame, (1280, 720))
        _, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


if __name__ == '__main__':
    app.run(debug=True)
