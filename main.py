import os
import cv2
import time
import sqlite3
import datetime
import threading
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from mpl_toolkits.mplot3d import Axes3D
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import telebot
from telebot import types

# ============================
# НАСТРОЙКИ
# ============================

TOKEN = os.getenv("TELEGRAM_TOKEN", "ВСТАВЬ_СВОЙ_ТОКЕН_БОТА")
bot = telebot.TeleBot(TOKEN)

DB_PATH = "throws.db"
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# YOLOv8m-pose
model = YOLO("yolov8m-pose.pt")

# ============================
# БАЗА ДАННЫХ
# ============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS throws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            peak_speed REAL,
            release_angle REAL,
            backswing REAL,
            follow REAL,
            head_stability REAL,
            pdc_profile TEXT,
            score REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            user_id INTEGER PRIMARY KEY,
            target_speed REAL,
            target_head_stab REAL,
            target_release_angle REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_throw(user_id, metrics, best_player, score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO throws (user_id, date, peak_speed, release_angle, backswing, follow, head_stability, pdc_profile, score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        datetime.datetime.now().strftime("%Y-%m-%d"),
        metrics["peak_speed"],
        metrics["release_angle"],
        metrics["backswing_length"],
        metrics["follow_length"],
        metrics["head_stability"],
        best_player,
        score
    ))
    conn.commit()
    conn.close()

def get_progress(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT date, peak_speed, release_angle, head_stability, score
        FROM throws
        WHERE user_id=?
        ORDER BY date
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def set_goals(user_id, target_speed, target_head_stab, target_release_angle):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO goals (user_id, target_speed, target_head_stab, target_release_angle)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            target_speed=excluded.target_speed,
            target_head_stab=excluded.target_head_stab,
            target_release_angle=excluded.target_release_angle
    """, (user_id, target_speed, target_head_stab, target_release_angle))
    conn.commit()
    conn.close()

def get_goals(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT target_speed, target_head_stab, target_release_angle
        FROM goals
        WHERE user_id=?
    """, (user_id,))
    row = c.fetchone()
    conn.close()
    return row

# ============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ АНАЛИЗА
# ============================

def compute_speed(path, fps):
    v = np.gradient(path, axis=0) * fps
    return np.linalg.norm(v, axis=1), v

def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def to3(x):
    return np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)

def smooth_signal(x, k=5):
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, x)

def stability_metric(points):
    center = np.mean(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    return {
        "mean_radius": float(np.mean(dist)),
        "max_radius": float(np.max(dist))
    }

def find_peaks(signal, min_distance=10, threshold_ratio=0.4):
    sig = np.array(signal)
    thr = threshold_ratio * np.max(sig)
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > thr:
            if len(peaks) == 0 or i - peaks[-1] > min_distance:
                peaks.append(i)
    return peaks

# ============================
# PDC ПРОФИЛИ
# ============================

pdc_profiles = {
    "Luke Littler": {
        "backswing": "средний",
        "smoothness": "очень высокая",
        "release_angle": 102,
        "speed": "очень высокая",
        "follow": "длинный",
        "head_stability": "высокая"
    },
    "Michael van Gerwen": {
        "backswing": "короткий",
        "smoothness": "высокая",
        "release_angle": 95,
        "speed": "высокая",
        "follow": "короткий",
        "head_stability": "высокая"
    },
    "Gerwyn Price": {
        "backswing": "средний",
        "smoothness": "средняя",
        "release_angle": 100,
        "speed": "очень высокая",
        "follow": "короткий",
        "head_stability": "очень высокая"
    }
}

def classify_length(v):
    return "короткий" if v < 15 else ("средний" if v < 40 else "длинный")

def classify_speed(v):
    return "средняя" if v < 150 else ("высокая" if v < 300 else "очень высокая")

def classify_smooth(v):
    return "очень высокая" if v < 2 else ("высокая" if v < 5 else "средняя")

def classify_follow(v):
    return "короткий" if v < 20 else ("средний" if v < 50 else "длинный")

def classify_head_stab(v):
    return "очень высокая" if v < 2 else ("высокая" if v < 5 else "средняя")

def throw_score(metrics):
    score = 0
    score += min(metrics["peak_speed"] / 250, 1.2) * 25
    score += max(0, (5 - metrics["head_stability"])) * 10
    score += max(0, 1 - abs(metrics["release_angle"] - 100) / 20) * 15
    score += min(metrics["follow_length"] / 40, 1.0) * 10
    return float(max(0, min(100, score)))

# ============================
# АНАЛИЗ ОДНОГО БРОСКА
# ============================

def analyze_single_throw(wrist_s, elbow_arr, shoulder_arr, head_arr, speed, acc, fps, center_idx, window_sec=1.2):
    half = int(window_sec * fps / 2)
    start = max(0, center_idx - half)
    end = min(len(speed) - 1, center_idx + half)

    w = wrist_s[start:end]
    el = elbow_arr[start:end]
    sh = shoulder_arr[start:end]
    hd = head_arr[start:end]
    sp = speed[start:end]
    ac = acc[start:end]

    ang_elbow = compute_angle(to3(sh), to3(el), to3(w))
    ang_shoulder = compute_angle(to3(hd), to3(sh), to3(el))

    rel = int(np.argmax(sp))

    if rel > 5:
        bs = int(np.argmin(sp[:rel]))
    else:
        bs = 0
    fs = bs
    fe = rel
    re = rel
    thr = 0.2 * np.max(sp)
    fo = len(sp) - 1
    for i in range(rel, len(sp)):
        if sp[i] < thr:
            fo = i
            break

    head_stab = stability_metric(hd)

    backswing_len = float(np.linalg.norm(w[bs] - w[0]))
    follow_len = float(np.linalg.norm(w[fo] - w[rel]))
    peak_speed = float(np.max(sp))
    release_angle = float(ang_elbow[rel])
    head_mean = head_stab["mean_radius"]
    smooth_val = float(np.mean(np.abs(ac[:bs]))) if bs > 1 else float(np.mean(np.abs(ac)))

    metrics = {
        "backswing_length": backswing_len,
        "follow_length": follow_len,
        "peak_speed": peak_speed,
        "release_angle": release_angle,
        "head_stability": head_mean,
        "smoothness": smooth_val,
        "elbow_angle_release": release_angle,
        "shoulder_angle_release": float(ang_shoulder[rel]),
        "backswing_frames": bs,
        "follow_frames": fo - rel
    }

    return {
        "w": w,
        "el": el,
        "sh": sh,
        "hd": hd,
        "sp": sp,
        "ac": ac,
        "ang_elbow": ang_elbow,
        "ang_shoulder": ang_shoulder,
        "rel": rel,
        "bs": bs,
        "fs": fs,
        "fe": fe,
        "re": re,
        "fo": fo,
        "head_stab": head_stab,
        "metrics": metrics
    }

def compare_with_pdc(metrics, head_stab):
    q = {
        "backswing": {
            "length": metrics["backswing_length"],
            "smoothness": metrics["smoothness"],
            "head_stability": head_stab["mean_radius"]
        },
        "release": {
            "speed": metrics["peak_speed"],
            "elbow_angle": metrics["elbow_angle_release"]
        },
        "follow_through": {
            "length": metrics["follow_length"]
        }
    }

    your = {
        "backswing": classify_length(q["backswing"]["length"]),
        "smoothness": classify_smooth(q["backswing"]["smoothness"]),
        "release_angle": metrics["release_angle"],
        "speed": classify_speed(q["release"]["speed"]),
        "follow": classify_follow(q["follow_through"]["length"]),
        "head_stability": classify_head_stab(q["backswing"]["head_stability"])
    }

    def score(y, r):
        s = 0
        s += y["backswing"] == r["backswing"]
        s += y["smoothness"] == r["smoothness"]
        s += y["speed"] == r["speed"]
        s += y["follow"] == r["follow"]
        s += y["head_stability"] == r["head_stability"]
        s += abs(y["release_angle"] - r["release_angle"]) < 8
        return s

    scores = {n: score(your, p) for n, p in pdc_profiles.items()}
    best = max(scores, key=scores.get)
    return your, scores, best

# ============================
# ГРАФИКИ И ОТЧЁТЫ
# ============================

def generate_plots(user_id, throw_data_list):
    plots = {}
    best_idx = int(np.argmax([np.max(t["sp"]) for t in throw_data_list]))
    t = throw_data_list[best_idx]

    speed_plot = f"{user_id}_speed.png"
    plt.figure(figsize=(8, 3))
    plt.plot(t["sp"], label="Скорость кисти")
    plt.axvspan(0, t["bs"], alpha=.2, label="Backswing")
    plt.axvspan(t["fs"], t["fe"], alpha=.2, label="Forward")
    plt.axvline(t["rel"], color='r', linestyle='--', label="Release")
    plt.axvspan(t["re"], t["fo"], alpha=.2, label="Follow-through")
    plt.grid()
    plt.legend()
    plt.title("Скорость кисти и фазы")
    plt.tight_layout()
    plt.savefig(speed_plot)
    plt.close()
    plots["speed"] = speed_plot

    elbow_plot = f"{user_id}_elbow.png"
    plt.figure(figsize=(8, 3))
    plt.plot(t["ang_elbow"], label="Угол в локте")
    plt.axvline(t["rel"], color='r', linestyle='--', label="Release")
    plt.grid()
    plt.legend()
    plt.title("Угол в локте по кадрам")
    plt.tight_layout()
    plt.savefig(elbow_plot)
    plt.close()
    plots["elbow"] = elbow_plot

    traj_plot = f"{user_id}_traj.png"
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    w = t["w"]
    ax.plot(w[:, 0], w[:, 1], np.zeros_like(w[:, 0]))
    ax.scatter(w[t["rel"], 0], w[t["rel"], 1], 0, color='r', label="Release")
    ax.set_title("3D-траектория кисти (XY)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(traj_plot)
    plt.close()
    plots["traj"] = traj_plot

    all_w = np.concatenate([td["w"] for td in throw_data_list], axis=0)
    heat_plot = f"{user_id}_heat.png"
    plt.figure(figsize=(5, 4))
    plt.hist2d(all_w[:, 0], all_w[:, 1], bins=40, cmap='hot')
    plt.colorbar(label="Плотность")
    plt.title("Тепловая карта движения кисти")
    plt.tight_layout()
    plt.savefig(heat_plot)
    plt.close()
    plots["heat"] = heat_plot

    return plots, best_idx

def generate_pdf(user_id, metrics, best_player, your_profile, score):
    pdf_name = f"{user_id}_report.pdf"
    date = datetime.datetime.now().strftime("%d.%m.%Y")

    c = canvas.Canvas(pdf_name, pagesize=A4)
    w_page, h_page = A4
    y = h_page - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Отчёт по технике броска")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Дата: {date}")
    y -= 20
    c.drawString(50, y, f"PDC-профиль: {best_player}")
    y -= 20
    c.drawString(50, y, f"Оценка техники: {score:.1f} / 100")
    y -= 20

    c.drawString(50, y, f"Пиковая скорость: {metrics['peak_speed']:.1f}")
    y -= 15
    c.drawString(50, y, f"Угол релиза (локоть): {metrics['elbow_angle_release']:.1f}°")
    y -= 15
    c.drawString(50, y, f"Backswing длина: {metrics['backswing_length']:.1f}")
    y -= 15
    c.drawString(50, y, f"Follow-through длина: {metrics['follow_length']:.1f}")
    y -= 15
    c.drawString(50, y, f"Стабильность головы: {metrics['head_stability']:.3f}")
    y -= 25

    c.drawString(50, y, "Классификация броска:")
    y -= 15
    c.drawString(60, y, f"Backswing: {your_profile['backswing']}")
    y -= 15
    c.drawString(60, y, f"Плавность: {your_profile['smoothness']}")
    y -= 15
    c.drawString(60, y, f"Скорость: {your_profile['speed']}")
    y -= 15
    c.drawString(60, y, f"Follow-through: {your_profile['follow']}")
    y -= 15
    c.drawString(60, y, f"Стабильность головы: {your_profile['head_stability']}")
    y -= 15
    c.drawString(60, y, f"Угол релиза: {your_profile['release_angle']:.1f}°")

    c.showPage()
    c.save()
    return pdf_name

def build_training_plan(metrics, your_profile, best_player):
    recs = []

    if metrics["peak_speed"] < 180:
        recs.append("Скорость: сделай 3×20 бросков с акцентом на ускорении руки в финальной фазе.")
    if metrics["head_stability"] > 5:
        recs.append("Стабильность головы: броски перед зеркалом, контролируя неподвижность головы.")
    if metrics["follow_length"] < 20:
        recs.append("Follow-through: 30 бросков с акцентом на полном завершении движения руки.")
    if abs(metrics["release_angle"] - 100) > 15:
        recs.append("Угол релиза: медленные броски с фиксацией позиции руки в момент выпуска.")

    if not recs:
        recs.append("Техника выглядит сбалансированной. Продолжай тренироваться в том же режиме.")

    return recs

def check_goals_text(user_id, metrics):
    goals = get_goals(user_id)
    if not goals:
        return None
    ts, th, ta = goals
    text = "Сравнение с твоими целями:\n"
    text += f"• Скорость: {metrics['peak_speed']:.1f} / цель {ts:.1f}\n"
    text += f"• Стабильность головы: {metrics['head_stability']:.3f} / цель {th:.3f} (меньше — лучше)\n"
    text += f"• Угол релиза: {metrics['release_angle']:.1f}° / цель {ta:.1f}°\n"
    return text

# ============================
# ОСНОВНОЙ АНАЛИЗ ВИДЕО
# ============================

def analyze_video(video_path, user_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    wrist_list, elbow_list, shoulder_list, head_list = [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, verbose=False)
        if len(results) > 0 and len(results[0].keypoints) > 0:
            kps = results[0].keypoints.xy[0].cpu().numpy()
            head_list.append(kps[0])
            shoulder_list.append(kps[5])
            elbow_list.append(kps[7])
            wrist_list.append(kps[9])

    cap.release()

    wrist_arr = np.array(wrist_list)
    elbow_arr = np.array(elbow_list)
    shoulder_arr = np.array(shoulder_list)
    head_arr = np.array(head_list)

    if len(wrist_arr) < 15:
        raise RuntimeError("Слишком мало кадров с позой для анализа.")

    wrist_s = smooth_signal(wrist_arr, k=7)
    speed, v = compute_speed(wrist_s, fps=fps)
    acc = np.gradient(speed)

    peaks = find_peaks(speed, min_distance=int(fps * 0.5), threshold_ratio=0.4)
    if not peaks:
        peaks = [int(np.argmax(speed))]

    throw_data_list = []
    for p in peaks:
        td = analyze_single_throw(wrist_s, elbow_arr, shoulder_arr, head_arr, speed, acc, fps, p)
        throw_data_list.append(td)

    plots, best_idx = generate_plots(user_id, throw_data_list)
    best_throw = throw_data_list[best_idx]
    your_profile, scores, best_player = compare_with_pdc(best_throw["metrics"], best_throw["head_stab"])
    score = throw_score(best_throw["metrics"])
    pdf_report = generate_pdf(user_id, best_throw["metrics"], best_player, your_profile, score)

    save_throw(user_id, best_throw["metrics"], best_player, score)

    training_plan = build_training_plan(best_throw["metrics"], your_profile, best_player)
    goals_text = check_goals_text(user_id, best_throw["metrics"])

    return {
        "plots": plots,
        "best_idx": best_idx,
        "your_profile": your_profile,
        "scores": scores,
        "best_player": best_player,
        "pdf": pdf_report,
        "score": score,
        "metrics": best_throw["metrics"],
        "training_plan": training_plan,
        "goals_text": goals_text
    }

# ============================
# TELEGRAM-БОТ
# ============================

@bot.message_handler(commands=['start', 'help'])
def start_cmd(message):
    text = (
        "Привет! Я AI Dart Coach.\n\n"
        "Отправь мне видео своего броска — я проанализирую технику, "
        "сравню с PDC и дам оценку.\n\n"
        "Команды:\n"
        "/setgoal скорость стабильность угол — задать цели\n"
        "/progress — посмотреть прогресс\n"
    )
    bot.reply_to(message, text)

@bot.message_handler(commands=['setgoal'])
def setgoal_cmd(message):
    try:
        _, s, h, a = message.text.split()
        s, h, a = float(s), float(h), float(a)
    except Exception:
        bot.reply_to(message, "Формат: /setgoal <скорость> <стабильность> <угол>\nПример: /setgoal 250 2.0 100")
        return
    set_goals(message.from_user.id, s, h, a)
    bot.reply_to(message, "Цели обновлены.")

@bot.message_handler(commands=['progress'])
def progress_cmd(message):
    user_id = message.from_user.id
    rows = get_progress(user_id)
    if len(rows) < 2:
        bot.reply_to(message, "Недостаточно данных для анализа прогресса. Нужно минимум 2 броска.")
        return

    dates = [r[0] for r in rows]
    speeds = [r[1] for r in rows]
    angles = [r[2] for r in rows]
    stability = [r[3] for r in rows]
    scores = [r[4] for r in rows]

    speed_plot = f"{user_id}_progress_speed.png"
    plt.figure(figsize=(6,3))
    plt.plot(dates, speeds, marker='o')
    plt.title("Прогресс скорости")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(speed_plot)
    plt.close()

    angle_plot = f"{user_id}_progress_angle.png"
    plt.figure(figsize=(6,3))
    plt.plot(dates, angles, marker='o')
    plt.title("Прогресс угла релиза")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(angle_plot)
    plt.close()

    stab_plot = f"{user_id}_progress_stab.png"
    plt.figure(figsize=(6,3))
    plt.plot(dates, stability, marker='o')
    plt.title("Прогресс стабильности головы")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(stab_plot)
    plt.close()

    score_plot = f"{user_id}_progress_score.png"
    plt.figure(figsize=(6,3))
    plt.plot(dates, scores, marker='o')
    plt.title("Прогресс оценки техники")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(score_plot)
    plt.close()

    bot.send_message(user_id, "Твой прогресс:")
    for p in [speed_plot, angle_plot, stab_plot, score_plot]:
        with open(p, "rb") as f:
            bot.send_photo(user_id, f)

    trend_speed = speeds[-1] - speeds[0]
    trend_score = scores[-1] - scores[0]
    text = "Анализ прогресса:\n"
    text += f"• Скорость: {'+' if trend_speed>=0 else ''}{trend_speed:.1f}\n"
    text += f"• Оценка техники: {'+' if trend_score>=0 else ''}{trend_score:.1f}\n"
    bot.send_message(user_id, text)

    for p in [speed_plot, angle_plot, stab_plot, score_plot]:
        if os.path.exists(p):
            os.remove(p)

@bot.message_handler(content_types=['video'])
def handle_video(message):
    user_id = message.from_user.id
    file_info = bot.get_file(message.video.file_id)
    downloaded = bot.download_file(file_info.file_path)

    filename = os.path.join(VIDEO_DIR, f"{user_id}_{int(time.time())}.mp4")
    with open(filename, "wb") as f:
        f.write(downloaded)

    bot.reply_to(message, "Видео получено. Анализирую бросок...")

    def worker():
        try:
            result = analyze_video(filename, user_id)
        except Exception as e:
            bot.send_message(user_id, f"Ошибка анализа: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            return

        summary = (
            f"Готово!\n\n"
            f"PDC-профиль: {result['best_player']}\n"
            f"Оценка техники: {result['score']:.1f} / 100\n"
            f"Пиковая скорость: {result['metrics']['peak_speed']:.1f}\n"
            f"Угол релиза: {result['metrics']['release_angle']:.1f}°\n"
        )
        bot.send_message(user_id, summary)

        if result["goals_text"]:
            bot.send_message(user_id, result["goals_text"])

        with open(result["pdf"], "rb") as f:
            bot.send_document(user_id, f)

        markup = types.InlineKeyboardMarkup()
        btn = types.InlineKeyboardButton("Показать полный анализ", callback_data=f"full_{user_id}")
        markup.add(btn)
        bot.send_message(user_id, "Хочешь увидеть полный анализ?", reply_markup=markup)

        if os.path.exists(filename):
            os.remove(filename)

    threading.Thread(target=worker).start()

@bot.callback_query_handler(func=lambda call: call.data.startswith("full_"))
def callback_full(call):
    user_id = call.from_user.id
    plots = [f"{user_id}_speed.png", f"{user_id}_elbow.png", f"{user_id}_traj.png", f"{user_id}_heat.png"]
    for p in plots:
        if os.path.exists(p):
            with open(p, "rb") as f:
                bot.send_photo(user_id, f)
            os.remove(p)
        else:
            bot.send_message(user_id, "Детальные графики уже были удалены или ещё не созданы.")

# ============================
# СТАРТ БОТА
# ============================

if __name__ == "__main__":
    print("AI Dart Coach бот запущен (long polling)...")
    bot.infinity_polling()
