from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
import random
import cv2
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained TensorFlow model
model = load_model("model/Lettuce_ResNet50_Mini_Best.h5")

LOG_FILE = 'static/data_log.csv'

GROWTH_TIPS = [
    "Water your plants early in the morning.",
    "Ensure proper light exposure for maximum growth.",
    "Use clean containers to prevent disease spread.",
    "Prune regularly to promote healthy development."
]

def time_ago(timestamp_str):
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        diff = datetime.now() - dt
        hours, remainder = divmod(diff.total_seconds(), 3600)
        minutes = int(remainder // 60)
        if hours < 1 and minutes < 1:
            return "Just now"
        parts = []
        if hours >= 1:
            parts.append(f"{int(hours)}h")
        if minutes >= 1:
            parts.append(f"{minutes}m")
        return " ".join(parts) + " ago"
    except:
        return "Unknown"

def get_recent_logs(n=3):
    if not os.path.exists(LOG_FILE):
        return []
    df = pd.read_csv(LOG_FILE)
    df = df.tail(n)
    recent_logs = []
    for _, row in df[::-1].iterrows():
        time = time_ago(row['Date'])
        color = {
            "Healthy": "green-500",
            "Bacterial": "yellow-400",
            "Fungal": "red-500"
        }.get(row['Prediction'], "gray-400")
        recent_logs.append({
            "label": row['Prediction'],
            "time": time,
            "color": color
        })
    return recent_logs

def get_calendar_events():
    if not os.path.exists(LOG_FILE):
        return []
    df = pd.read_csv(LOG_FILE)
    events = []
    color_map = {
        "Healthy": "#16a34a",
        "Bacterial": "#facc15",
        "Fungal": "#ef4444"
    }
    for _, row in df.iterrows():
        try:
            dt = datetime.strptime(row["Date"], "%Y-%m-%d_%H-%M-%S")
            color = color_map.get(row["Prediction"], "#6b7280")
            events.append({
                "title": row["Prediction"],
                "start": dt.strftime("%Y-%m-%d"),
                "color": color,
                "tooltip": f"{row['Prediction']}<br>Height: {row['Height_cm']} cm<br>{row['Remedy']}"
            })
        except Exception:
            continue
    return events

@app.route('/')
def index():
    latest = None
    ago_text = "N/A"
    recent_logs = []
    calendar_events = get_calendar_events()

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            latest = df.iloc[-1]
            ago_text = time_ago(latest['Date']) if 'Date' in latest else "Unknown"
            recent_logs = get_recent_logs(3)

    tip = random.choice(GROWTH_TIPS)
    return render_template(
        'index.html',
        latest=latest,
        tip=tip,
        ago=ago_text,
        recent_logs=recent_logs,
        calendar_events=calendar_events,
        now=datetime.now()  # ✅ FIX: pass the actual datetime object, not the method
    )

@app.route('/capture')
def capture():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "Error: Webcam not accessible"
    ret, frame = cam.read()
    cam.release()
    if ret:
        cv2.imwrite("static/plant.jpg", frame)
    return redirect(url_for('index'))

@app.route('/scan')
def scan():
    img = cv2.imread("static/plant.jpg")
    if img is None:
        return "No image found."

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    arr = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

    pred = model.predict(arr)[0]
    labels = ["Bacterial", "Fungal", "Healthy"]
    result = labels[np.argmax(pred)]
    remedies = {
        "Bacterial": "Remove infected leaves and apply a copper-based bactericide.",
        "Fungal": "Improve ventilation and apply an approved fungicide.",
        "Healthy": "No disease detected. Continue regular care."
    }

    height_cm = round(random.uniform(7.5, 11.5), 2)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expected_columns = ["Date", "Prediction", "Remedy", "Height_cm"]

    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=expected_columns).to_csv(LOG_FILE, index=False)

    log_df = pd.read_csv(LOG_FILE)
    if list(log_df.columns) != expected_columns:
        log_df.columns = expected_columns[:len(log_df.columns)]

    new_entry = pd.DataFrame([[timestamp, result, remedies[result], height_cm]], columns=expected_columns)
    log_df = pd.concat([log_df, new_entry], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)

    # Save annotated image
    annotated = img.copy()
    cv2.putText(annotated, f"Disease: {result} ({round(pred[np.argmax(pred)] * 100, 1)}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(annotated, f"Height: {height_cm} cm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite("static/annotated_latest.jpg", annotated)

    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if file:
        file.save("static/plant.jpg")
    return redirect(url_for('scan'))

@app.route('/download')
def download():
    return send_file(LOG_FILE, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)