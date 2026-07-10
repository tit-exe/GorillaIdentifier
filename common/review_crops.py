# review_crops.py
# universal crop reviewer
# 
#
# Drag and drop ANY crop image onto the window.
# Works with known gorilla individuals, wild crops, or any mix.
# Finds the crop entry in data/crops.json regardless of version (V1–V4).
#
# KEYS:
#   D      = draw new box (click+drag, does NOT auto-save)
#   Enter  = validate + save + next
#   R      = reject (JSON only, files kept)
#   S      = skip
#   A / <- = previous
#   Z      = reset box to YOLO original
#   Del/X  = delete crop + source photo + JSON entry
#   Esc    = cancel draw mode
#   Q      = quit
#
# RUN:
#   python common/review_crops.py
#   python common/review_crops.py path/to/crop1.jpg path/to/crop2.jpg
#   python common/review_crops.py --json path/to/other.json

import sys
import json
import shutil
import threading
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont,
    QBrush, QDragEnterEvent, QDropEvent, QPaintEvent, QMouseEvent
)

# ==============================================================================
# CONFIG — import paths from config_loader so this script has zero hardcoded paths
# ==============================================================================

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from common.config_loader import CROPS_JSON, REPO_ROOT, resolve_path, to_relative

CROP_SIZE = 224
HANDLE_R  = 7

# ==============================================================================
# JSON MANAGER — atomic write + self-healing backup
# ==============================================================================

class JsonManager:
    def __init__(self, path):
        self.path  = path
        self._lock = threading.Lock()
        self._data = {}

    def load(self):
        with self._lock:
            if self.path.exists():
                try:
                    with open(self.path, encoding="utf-8") as f:
                        self._data = json.load(f)
                    print(f"[INFO] JSON loaded: {len(self._data):,} entries from {self.path}")
                    return dict(self._data)
                except json.JSONDecodeError:
                    print("[WARN] Main JSON corrupted, trying backup...")
            bak = self.path.with_suffix(".bak")
            if bak.exists():
                try:
                    with open(bak, encoding="utf-8") as f:
                        self._data = json.load(f)
                    shutil.copy2(bak, self.path)
                    print(f"[WARN] Loaded from backup: {len(self._data):,} entries")
                    return dict(self._data)
                except Exception:
                    pass
            self._data = {}
            print(f"[WARN] No JSON found at {self.path} - starting empty")
            return {}

    def save(self, data):
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            bak = self.path.with_suffix(".bak")
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                if self.path.exists():
                    shutil.copy2(self.path, bak)
                tmp.replace(self.path)
                self._data = data
                return True
            except Exception as e:
                print(f"[ERROR] Save failed: {e}")
                if tmp.exists():
                    try: tmp.unlink()
                    except: pass
                return False

    def get_data(self):
        with self._lock:
            return dict(self._data)

# ==============================================================================
# FIND ENTRY — 3-level lookup, works for known/wild/any version
# ==============================================================================

def find_entry(crop_path: Path, db: dict):
    """
    Find the JSON entry for a given crop file.

    Lookup order:
      1. Direct key "IndividualName/stem"  — known individuals
      2. Direct key "wild/stem"            — wild/background crops
      3. Scan all entries for matching crop_file basename  — any structure
      4. Match by individu+stem fields     — legacy entries
    """
    name     = crop_path.name
    stem     = crop_path.stem
    individu = crop_path.parent.name   # folder name = individual or "wild"

    # 1. Direct known: "Molly/photo123"
    key = f"{individu}/{stem}"
    if key in db:
        return key, db[key]

    # 2. Wild crops: "wild/<stem>"  (works even if dropped from a non-wild folder)
    wild_key = f"wild/{stem}"
    if wild_key in db:
        return wild_key, db[wild_key]

    # 3. Scan on crop_file basename (works for any path structure)
    for k, v in db.items():
        if not isinstance(v, dict):
            continue
        cf = v.get("crop_file", "")
        if cf and Path(cf).name == name:
            return k, v

    # 4. Match individu + stem fields (legacy entries with different key format)
    for k, v in db.items():
        if not isinstance(v, dict):
            continue
        if v.get("individu") == individu and v.get("stem") == stem:
            return k, v

    return None, {}

# ==============================================================================
# RESOLVE PHOTO SOURCE — handles relative and absolute paths
# ==============================================================================

def resolve_photo(entry: dict) -> Path | None:
    """
    Resolve the photo_source field from a JSON entry.
    Handles:
      - Relative paths (relative to REPO_ROOT) — new format
      - Absolute paths                          — legacy format
    Returns the resolved Path if it exists, else None.
    """
    src = entry.get("photo_source", "")
    if not src:
        return None
    p = Path(src)
    if p.is_absolute():
        return p if p.exists() else None
    # Relative: resolve from REPO_ROOT
    resolved = (REPO_ROOT / p).resolve()
    return resolved if resolved.exists() else None

# ==============================================================================
# CROP REGEN — regenerate 224x224 crop from original photo and bounding box
# ==============================================================================

def regen_crop(photo_src, x1, y1, x2, y2, dest):
    img = cv2.imread(str(photo_src))
    if img is None: return None
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(x1+1, min(int(x2), w))
    y2 = max(y1+1, min(int(y2), h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None
    out = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)
    if dest is not None:
        cv2.imwrite(str(dest), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out

# ==============================================================================
# IMAGE CANVAS — shows original photo with interactive bounding box
# ==============================================================================

class ImageCanvas(QWidget):
    boxChanged = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self._pixmap         = None
        self._bx1 = self._by1 = 0
        self._bx2 = self._by2 = 100
        self._orig_box       = (0, 0, 100, 100)
        self._draw_mode      = False
        self._drag_handle    = None
        self._drag_start_w   = None
        self._drag_start_box = None
        self._drawing        = False
        self._draw_p1        = None

    def set_image_and_box(self, img_bgr, box):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self._pixmap    = QPixmap.fromImage(qimg)
        self._orig_box  = tuple(box)
        self.set_box(*box)
        self._draw_mode = False
        self._drawing   = False
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_box(self, x1, y1, x2, y2):
        if not self._pixmap: return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        self._bx1 = max(0, min(int(x1), pw-1))
        self._by1 = max(0, min(int(y1), ph-1))
        self._bx2 = max(self._bx1+1, min(int(x2), pw))
        self._by2 = max(self._by1+1, min(int(y2), ph))
        self.boxChanged.emit(self._bx1, self._by1, self._bx2, self._by2)
        self.update()

    def get_box(self):   return (self._bx1, self._by1, self._bx2, self._by2)
    def reset_box(self): self.set_box(*self._orig_box)

    def set_draw_mode(self, on):
        self._draw_mode = on
        self._drawing   = False
        self._draw_p1   = None
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        self.update()

    def clear(self):
        self._pixmap = None
        self._draw_mode = False
        self.update()

    def _fit_rect(self):
        if not self._pixmap: return QRect()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        s = min(ww/pw, wh/ph)
        nw, nh = int(pw*s), int(ph*s)
        return QRect((ww-nw)//2, (wh-nh)//2, nw, nh)

    def _to_widget(self, ix, iy, r):
        if not self._pixmap or r.isEmpty(): return QPoint(int(ix), int(iy))
        sx = r.width()  / self._pixmap.width()
        sy = r.height() / self._pixmap.height()
        return QPoint(r.x() + int(ix*sx), r.y() + int(iy*sy))

    def _to_image(self, wx, wy, r):
        if not self._pixmap or r.isEmpty(): return (wx, wy)
        sx = self._pixmap.width()  / r.width()
        sy = self._pixmap.height() / r.height()
        return (max(0.0, min((wx-r.x())*sx, float(self._pixmap.width()))),
                max(0.0, min((wy-r.y())*sy, float(self._pixmap.height()))))

    def _handles(self, r):
        p = lambda x, y: self._to_widget(x, y, r)
        x1, y1, x2, y2 = self._bx1, self._by1, self._bx2, self._by2
        mx, my = (x1+x2)/2, (y1+y2)/2
        return {
            "tl": p(x1,y1), "tm": p(mx,y1), "tr": p(x2,y1),
            "rm": p(x2,my), "br": p(x2,y2), "bm": p(mx,y2),
            "bl": p(x1,y2), "lm": p(x1,my),
        }

    def _hit(self, wx, wy, r):
        for name, pt in self._handles(r).items():
            if abs(wx-pt.x()) <= HANDLE_R and abs(wy-pt.y()) <= HANDLE_R:
                return name
        p1 = self._to_widget(self._bx1, self._by1, r)
        p2 = self._to_widget(self._bx2, self._by2, r)
        if QRect(p1, p2).normalized().contains(QPoint(wx, wy)):
            return "move"
        return None

    def mousePressEvent(self, e):
        if not self._pixmap: return
        r  = self._fit_rect()
        wx, wy = e.x(), e.y()
        if e.button() == Qt.RightButton:
            ix, iy = self._to_image(wx, wy, r)
            hw = (self._bx2-self._bx1)/2
            hh = (self._by2-self._by1)/2
            self.set_box(ix-hw, iy-hh, ix+hw, iy+hh)
            return
        if e.button() != Qt.LeftButton: return
        if self._draw_mode:
            self._drawing = True
            self._draw_p1 = (wx, wy)
            return
        h = self._hit(wx, wy, r)
        if h:
            self._drag_handle    = h
            self._drag_start_w   = (wx, wy)
            self._drag_start_box = self.get_box()
            cursors = {
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "tm": Qt.SizeVerCursor,   "bm": Qt.SizeVerCursor,
                "lm": Qt.SizeHorCursor,   "rm": Qt.SizeHorCursor,
                "move": Qt.SizeAllCursor,
            }
            self.setCursor(cursors.get(h, Qt.ArrowCursor))

    def mouseMoveEvent(self, e):
        if not self._pixmap: return
        r  = self._fit_rect()
        wx, wy = e.x(), e.y()
        if not self._drag_handle and not self._drawing and not self._draw_mode:
            h = self._hit(wx, wy, r)
            cursors = {
                "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                "tm": Qt.SizeVerCursor,   "bm": Qt.SizeVerCursor,
                "lm": Qt.SizeHorCursor,   "rm": Qt.SizeHorCursor,
                "move": Qt.SizeAllCursor,
            }
            self.setCursor(cursors.get(h, Qt.ArrowCursor))
        if self._drawing and self._draw_p1:
            ix1, iy1 = self._to_image(min(self._draw_p1[0], wx),
                                       min(self._draw_p1[1], wy), r)
            ix2, iy2 = self._to_image(max(self._draw_p1[0], wx),
                                       max(self._draw_p1[1], wy), r)
            self.set_box(ix1, iy1, ix2, iy2)
            return
        if self._drag_handle and self._drag_start_w:
            sw, sh = self._drag_start_w
            if not r.isEmpty() and self._pixmap:
                dx = (wx-sw) * self._pixmap.width()  / r.width()
                dy = (wy-sh) * self._pixmap.height() / r.height()
            else:
                dx, dy = wx-sw, wy-sh
            b  = self._drag_start_box
            x1, y1, x2, y2 = b
            h  = self._drag_handle
            if   h == "move": x1+=dx; y1+=dy; x2+=dx; y2+=dy
            elif h == "tl":   x1+=dx; y1+=dy
            elif h == "tr":   x2+=dx; y1+=dy
            elif h == "bl":   x1+=dx; y2+=dy
            elif h == "br":   x2+=dx; y2+=dy
            elif h == "tm":   y1+=dy
            elif h == "bm":   y2+=dy
            elif h == "lm":   x1+=dx
            elif h == "rm":   x2+=dx
            self.set_box(x1, y1, x2, y2)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_handle    = None
            self._drag_start_w   = None
            self._drag_start_box = None
            if self._drawing:
                self._drawing   = False
                self._draw_p1   = None
                self._draw_mode = False
                self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 10, 10))
        if not self._pixmap:
            p.setPen(QColor(50, 50, 50))
            p.drawRect(self.rect().adjusted(30, 30, -30, -30))
            p.setPen(QColor(70, 70, 70))
            p.setFont(QFont("Consolas", 13))
            p.drawText(self.rect(), Qt.AlignCenter,
                "Drag & drop crop images here\n\n"
                "D = draw new box    Enter = save\n"
                "R = reject    S = skip    A = back\n"
                "Z = reset    Del/X = delete    Q = quit")
            return
        r   = self._fit_rect()
        p.drawPixmap(r, self._pixmap)
        pt1 = self._to_widget(self._bx1, self._by1, r)
        pt2 = self._to_widget(self._bx2, self._by2, r)
        bw  = self._bx2 - self._bx1
        bh  = self._by2 - self._by1
        ratio = bw/bh if bh > 0 else 0
        col = QColor(255, 60, 80) if ratio < 0.5 or ratio > 2.0 else QColor(0, 220, 90)
        p.setPen(QPen(col, 2))
        p.drawRect(QRect(pt1, pt2).normalized())
        p.setBrush(QBrush(col))
        p.setPen(QPen(Qt.white, 1))
        for pt in self._handles(r).values():
            p.drawEllipse(pt, HANDLE_R, HANDLE_R)
        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(col))
        p.drawText(pt1.x(), pt1.y() - 6,
                   f"{int(bw)}x{int(bh)}  ratio={ratio:.2f}")
        if self._draw_mode:
            p.setFont(QFont("Consolas", 11, QFont.Bold))
            p.setPen(QPen(QColor(255, 160, 0)))
            p.drawText(10, 24, "DRAW MODE - click and drag to draw a new box")

# ==============================================================================
# PREVIEW
# ==============================================================================

class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(224, 224)
        self.setStyleSheet("border:1px solid #333;background:#0a0a0a;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("preview")

    def set_crop(self, crop_bgr):
        if crop_bgr is None:
            self.setText("error")
            return
        rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))

# ==============================================================================
# MAIN WINDOW
# ==============================================================================

class ReviewWindow(QMainWindow):
    def __init__(self, json_path: Path, initial_files: list):
        super().__init__()
        self.json_mgr      = JsonManager(json_path)
        self.db            = self.json_mgr.load()
        self.queue         = deque()
        self.history       = []
        self.current_crop  = None
        self.current_key   = None
        self.current_entry = None
        self.current_img   = None
        self.current_photo = None   # resolved Path to original photo
        self.stats = {
            "corrected": 0, "validated": 0,
            "rejected":  0, "skipped":   0, "deleted": 0
        }
        self._build_ui()
        self.setAcceptDrops(True)
        self.setWindowTitle("Crop Reviewer  |")
        self.resize(1300, 820)
        if initial_files:
            self._enqueue([Path(f) for f in initial_files])

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content = QWidget()
        cl = QHBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        self.canvas = ImageCanvas()
        self.canvas.boxChanged.connect(self._on_box_changed)
        cl.addWidget(self.canvas, stretch=1)

        right = QFrame()
        right.setFixedWidth(260)
        right.setStyleSheet(
            "QFrame{background:#111;border-left:1px solid #222;}")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 12, 12, 12)
        rl.setSpacing(8)

        lbl = QLabel("PREVIEW  224x224")
        lbl.setStyleSheet(
            "color:#444;font-family:Consolas;font-size:10px;")
        rl.addWidget(lbl, alignment=Qt.AlignHCenter)

        self.preview = PreviewLabel()
        rl.addWidget(self.preview, alignment=Qt.AlignHCenter)

        self.lbl_box = QLabel("")
        self.lbl_box.setStyleSheet(
            "color:#666;font-family:Consolas;font-size:10px;"
            "background:#0d0d0d;padding:6px;border-radius:3px;")
        self.lbl_box.setWordWrap(True)
        rl.addWidget(self.lbl_box)
        rl.addStretch()

        def make_btn(text, shortcut, color, slot):
            b = QPushButton(f"{text}  [{shortcut}]")
            b.setFixedHeight(36)
            b.setStyleSheet(
                f"QPushButton{{background:{color};color:#eee;border:none;"
                f"border-radius:4px;font-family:Consolas;"
                f"font-size:12px;font-weight:bold;}}"
                f"QPushButton:hover{{opacity:0.8;}}"
            )
            b.clicked.connect(slot)
            rl.addWidget(b)
            return b

        make_btn("Draw new box", "D",     "#8a5c00", self._action_draw)
        make_btn("Validate",     "Enter", "#1a5c2e", self._action_validate)
        make_btn("Reject",       "R",     "#5c1a1a", self._action_reject)
        make_btn("Skip",         "S",     "#3a3a3a", self._action_skip)
        make_btn("Previous",     "A",     "#1a2a5c", self._action_previous)
        make_btn("Reset box",    "Z",     "#3a3a3a", self._action_reset)
        make_btn("DELETE ALL",   "Del/X", "#6e0000", self._action_delete)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet(
            "color:#555;font-family:Consolas;font-size:9px;")
        self.lbl_stats.setWordWrap(True)
        rl.addWidget(self.lbl_stats)

        cl.addWidget(right)
        root.addWidget(content, stretch=1)

        bot = QFrame()
        bot.setFixedHeight(28)
        bot.setStyleSheet(
            "QFrame{background:#0a0a0a;border-top:1px solid #1e1e1e;}")
        bl = QHBoxLayout(bot)
        bl.setContentsMargins(10, 0, 10, 0)
        self.lbl_status = QLabel(
            "Ready - drag crop images onto the window")
        self.lbl_status.setStyleSheet(
            "color:#555;font-family:Consolas;font-size:10px;")
        bl.addWidget(self.lbl_status)
        root.addWidget(bot)

    def _status(self, msg):
        self.lbl_status.setText(msg)

    def _update_stats(self):
        s = self.stats
        self.lbl_stats.setText(
            f"corrected: {s['corrected']}\n"
            f"validated: {s['validated']}\n"
            f"rejected:  {s['rejected']}\n"
            f"skipped:   {s['skipped']}\n"
            f"deleted:   {s['deleted']}\n"
            f"queue:     {len(self.queue)}"
        )

    def _on_box_changed(self, x1, y1, x2, y2):
        if self.current_img is None or self.current_photo is None:
            return
        crop = regen_crop(self.current_photo, x1, y1, x2, y2, dest=None)
        self.preview.set_crop(crop)
        bw   = x2 - x1
        bh   = y2 - y1
        ratio = bw/bh if bh > 0 else 0
        self.lbl_box.setText(
            f"x1={x1} y1={y1}\nx2={x2} y2={y2}\n"
            f"size: {bw}x{bh}px\nratio: {ratio:.2f}"
        )

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.exists():
                paths.append(p)
        if paths:
            self._enqueue(paths)

    def _enqueue(self, paths):
        for p in paths:
            self.queue.append(p)
        self._status(
            f"Added {len(paths)} file(s). Queue: {len(self.queue)}")
        self._update_stats()
        if self.current_crop is None:
            self._load_next()

    def _load_next(self):
        while self.queue:
            crop_path = self.queue.popleft()
            if self._load_crop(crop_path):
                self.history.append(crop_path)
                return
        self.current_crop = None
        self.canvas.clear()
        self.preview.setText("done")
        self._status("Queue empty - drop more files to continue")
        self._update_stats()

    def _load_crop(self, crop_path: Path) -> bool:
        self.db = self.json_mgr.get_data()
        key, entry = find_entry(crop_path, self.db)
        if not key:
            # No-JSON mode: show the crop directly (no original photo available)
            img = cv2.imread(str(crop_path))
            if img is None:
                self._status(f"[WARN] Cannot read crop: {crop_path.name} - skipping")
                return False
            self.current_crop  = crop_path
            self.current_key   = None
            self.current_entry = None
            self.current_img   = img
            self.current_photo = crop_path
            h, w = img.shape[:2]
            self.canvas.set_image_and_box(img, (0, 0, w, h))
            self._on_box_changed(0, 0, w, h)
            self._status(
                f"[no-JSON] {crop_path.parent.name}/{crop_path.name}  "
                f"Enter=OK  Del=supprimer  S=passer  queue={len(self.queue)}"
            )
            self._update_stats()
            return True

        # Resolve photo source — handles both relative and absolute paths
        photo = resolve_photo(entry)
        if photo is None:
            src_raw = entry.get("photo_source", "(none)")
            self._status(f"[WARN] Source photo not found: {src_raw} - skipping")
            return False

        img = cv2.imread(str(photo))
        if img is None:
            self._status(f"[WARN] Cannot read photo: {photo.name} - skipping")
            return False

        self.current_crop  = crop_path
        self.current_key   = key
        self.current_entry = entry
        self.current_img   = img
        self.current_photo = photo

        x1 = entry.get("crop_x1", 0)
        y1 = entry.get("crop_y1", 0)
        x2 = entry.get("crop_x2", 100)
        y2 = entry.get("crop_y2", 100)
        self.canvas.set_image_and_box(img, (x1, y1, x2, y2))
        self._on_box_changed(x1, y1, x2, y2)

        individu    = entry.get("individu") or "wild"
        conf        = entry.get("yolo_conf", "?")
        statut      = entry.get("statut", "?")
        source_type = entry.get("source_type", "")
        tag         = f"[{source_type}]" if source_type else ""
        self._status(
            f"{individu} / {crop_path.name}  {tag}  "
            f"conf={conf}   statut={statut}   "
            f"queue={len(self.queue)}"
        )
        self._update_stats()
        return True

    def _action_draw(self):
        if self.current_crop is None: return
        self.canvas.set_draw_mode(True)
        self._status("Draw mode: click+drag to draw a new box.  Esc=cancel")

    def _action_validate(self):
        if self.current_crop is None:
            return
        if self.current_entry is None:
            # No-JSON mode — just move to the next one
            self.stats["validated"] += 1
            self._status(f"OK: {self.current_crop.name}")
            self.current_crop = None
            self._load_next()
            return
        x1, y1, x2, y2 = self.canvas.get_box()

        # Regen crop from original photo with new box
        ok = regen_crop(self.current_photo, x1, y1, x2, y2, self.current_crop)
        if ok is None:
            self._status("[ERROR] Could not regenerate crop")
            return

        updated = dict(self.current_entry)
        updated.update({
            "crop_x1": x1, "crop_y1": y1,
            "crop_x2": x2, "crop_y2": y2,
            "statut":  "valide",
            "manually_reviewed": True,
            "review_date": datetime.now().isoformat(),
            # Store crop_file as relative path for portability
            "crop_file": to_relative(self.current_crop),
        })
        data = self.json_mgr.get_data()
        data[self.current_key] = updated
        self.json_mgr.save(data)
        self.stats["corrected"] += 1
        self._status(f"Saved: {self.current_crop.name}")
        self.current_crop = None
        self._load_next()

    def _action_reject(self):
        if self.current_crop is None or self.current_entry is None:
            return
        updated = dict(self.current_entry)
        updated["statut"]      = "rejete"
        updated["reject_date"] = datetime.now().isoformat()
        data = self.json_mgr.get_data()
        data[self.current_key] = updated
        self.json_mgr.save(data)
        self.stats["rejected"] += 1
        self._status(f"Rejected: {self.current_crop.name}")
        self.current_crop = None
        self._load_next()

    def _action_skip(self):
        if self.current_crop is None: return
        self.stats["skipped"] += 1
        self._status(f"Skipped: {self.current_crop.name}")
        self.current_crop = None
        self._load_next()

    def _action_previous(self):
        if len(self.history) < 2: return
        if self.current_crop:
            self.queue.appendleft(self.current_crop)
        prev = self.history[-2]
        self.history = self.history[:-2]
        self.queue.appendleft(prev)
        self.current_crop = None
        self._load_next()

    def _action_reset(self):
        if self.current_crop is None: return
        self.canvas.reset_box()

    def _action_delete(self):
        if self.current_crop is None:
            return
        if self.current_entry is None:
            # No-JSON mode — only delete the crop file
            if self.current_crop.exists():
                try:
                    self.current_crop.unlink()
                    self._status(f"Deleted: {self.current_crop.name}")
                except Exception as e:
                    self._status(f"[ERROR] Cannot delete: {e}")
            self.stats["deleted"] += 1
            self.current_crop = None
            self._load_next()
            return
        deleted = []
        # Delete crop file(s) — both recorded path and current drag path
        for f in [
            Path(self.current_entry.get("crop_file", "")),
            self.current_crop
        ]:
            if not f or not str(f): continue
            resolved = resolve_path(f) if not Path(str(f)).is_absolute() else Path(str(f))
            if resolved.exists():
                try:
                    resolved.unlink()
                    deleted.append(resolved.name)
                except Exception:
                    pass
        # Delete original photo (wild crops only — don't delete labeled individual photos)
        source_type = self.current_entry.get("source_type", "known")
        if source_type == "wild" and self.current_photo and self.current_photo.exists():
            try:
                self.current_photo.unlink()
                deleted.append(self.current_photo.name)
            except Exception:
                pass
        # Remove JSON entry
        data = self.json_mgr.get_data()
        if self.current_key in data:
            del data[self.current_key]
        self.json_mgr.save(data)
        self.stats["deleted"] += 1
        self._status(f"Deleted: {', '.join(deleted) if deleted else '(none found)'}")
        self.current_crop = None
        self._load_next()

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_D:
            self._action_draw()
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self._action_validate()
        elif k == Qt.Key_R:
            self._action_reject()
        elif k == Qt.Key_S:
            self._action_skip()
        elif k in (Qt.Key_A, Qt.Key_Left):
            self._action_previous()
        elif k == Qt.Key_Z:
            self._action_reset()
        elif k in (Qt.Key_Delete, Qt.Key_X):
            self._action_delete()
        elif k == Qt.Key_Escape:
            self.canvas.set_draw_mode(False)
            self._status("Draw mode cancelled")
        elif k == Qt.Key_Q:
            self.close()

    def closeEvent(self, e):
        s = self.stats
        print(
            f"\nSession: {s['corrected']} corrected  "
            f"{s['validated']} validated  {s['rejected']} rejected  "
            f"{s['skipped']} skipped  {s['deleted']} deleted"
        )
        e.accept()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crop reviewer — drag & drop any crop, works for all versions"
    )
    parser.add_argument(
        "--json", type=Path, default=CROPS_JSON,
        help=f"Path to crops JSON (default: data/crops.json)"
    )
    parser.add_argument(
        "files", nargs="*", type=str,
        help="Optional crop files to pre-load in the queue"
    )
    args = parser.parse_args()

    if not args.json.exists():
        print(f"[WARN] JSON not found: {args.json}")
        print("       Run first: python v1_megadesc_arcface/train.py --extract")

    initial_files = args.files

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt5.QtGui import QPalette
    pal = QPalette()
    pal.setColor(QPalette.Window,     QColor(12, 12, 12))
    pal.setColor(QPalette.WindowText, QColor(200, 200, 200))
    pal.setColor(QPalette.Base,       QColor(18, 18, 18))
    pal.setColor(QPalette.Button,     QColor(35, 35, 35))
    pal.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    app.setPalette(pal)

    win = ReviewWindow(args.json, initial_files)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
