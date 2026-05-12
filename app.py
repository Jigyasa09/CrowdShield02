# app.py — StampedeShield Main Server (Fixed)
# Fixes:
#   1. Camera stream: faster frame processing, proper MJPEG boundary
#   2. Video analysis: sequential frame read (no seek), skip per-frame anomaly,
#      streaming JSON-lines progress so frontend doesn't hang

import os, cv2, torch, json, time, threading, base64, io
import numpy as np
from flask import Flask, render_template, jsonify, request, Response, send_file, stream_with_context
from flask_socketio import SocketIO, emit
from collections import deque
from werkzeug.utils import secure_filename

from risk_calculator import (StampedeRiskCalculator, AlertManager,
                              quick_risk, RiskLevel)
from venue_graph import (create_stadium_graph, find_all_safe_exits,
                         update_edge_weights)

app      = Flask(__name__)
app.config['SECRET_KEY'] = 'stampedeshield2024'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)

ZONE_CONFIGS = [
    {'id':1,  'name':'North Stand A',   'capacity':800},
    {'id':2,  'name':'North Stand B',   'capacity':800},
    {'id':3,  'name':'North Stand C',   'capacity':800},
    {'id':4,  'name':'South Stand A',   'capacity':800},
    {'id':5,  'name':'South Stand B',   'capacity':800},
    {'id':6,  'name':'South Stand C',   'capacity':800},
    {'id':7,  'name':'East Stand',      'capacity':1200},
    {'id':8,  'name':'West Stand',      'capacity':1200},
    {'id':9,  'name':'North Concourse', 'capacity':400},
    {'id':10, 'name':'South Concourse', 'capacity':400},
    {'id':11, 'name':'East Concourse',  'capacity':400},
    {'id':12, 'name':'West Concourse',  'capacity':400},
    {'id':13, 'name':'North Tunnel L',  'capacity':150},
    {'id':14, 'name':'North Tunnel R',  'capacity':150},
]


# ── MODEL MANAGER ─────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.p2pnet        = None
        self.anomaly_model = None
        self.transform_vit = None
        self._load_models()

    def _load_models(self):
        try:
            from p2pnet_inference import P2PNetInference
            self.p2pnet = P2PNetInference(
                weights_path  = 'SHTechA.pth',
                backbone_path = 'checkpoints/vgg16_bn-6c64b313.pth'
            )
        except Exception as e:
            print(f"⚠️  P2PNet not loaded: {e} — using simulation")

        try:
            from vit_model import ViTAnomalyModel
            from torchvision import transforms
            self.anomaly_model = ViTAnomalyModel().to(DEVICE)
            self.anomaly_model.load_state_dict(
                torch.load('models/best_model.pt', map_location=DEVICE))
            self.anomaly_model.eval()
            self.transform_vit = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            print("✅ ViT anomaly model loaded")
        except Exception as e:
            print(f"⚠️  ViT not loaded: {e} — using simulation")

        loaded = []
        if self.p2pnet:        loaded.append("P2PNet")
        if self.anomaly_model: loaded.append("ViT")
        print(f"Models loaded: {loaded if loaded else ['simulation mode']}")

    def predict_crowd(self, frame_bgr):
        if self.p2pnet:
            return self.p2pnet.predict(frame_bgr)
        count = np.random.randint(50, 600)
        return count, [], self._sim_heatmap(frame_bgr, count)

    def predict_anomaly(self, frame_bgr) -> float:
        if self.anomaly_model is None:
            return float(np.random.beta(1, 5))
        img   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img, (224, 224))
        t     = self.transform_vit(img_r)
        clip  = torch.stack([t] * 16).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.anomaly_model(clip)
            probs  = torch.softmax(logits, dim=-1)
        return float(probs[0][1].cpu())

    def predict_anomaly_clip(self, frames_bgr) -> float:
        if self.anomaly_model is None or not frames_bgr:
            return float(np.random.beta(1, 5))
        n = len(frames_bgr)
        indices = np.linspace(0, n - 1, 16, dtype=int)
        sampled = [frames_bgr[i] for i in indices]
        tensors = []
        for f in sampled:
            img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            tensors.append(self.transform_vit(img))
        clip = torch.stack(tensors).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.anomaly_model(clip)
            probs  = torch.softmax(logits, dim=-1)
        return float(probs[0][1].cpu())

    def analyze_image_full(self, frame_bgr):
        count, points, heatmap = self.predict_crowd(frame_bgr)
        anomaly_score          = self.predict_anomaly(frame_bgr)
        annotated = self._annotate_frame(heatmap, count, anomaly_score)
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        heatmap_b64 = base64.b64encode(buf).decode('utf-8')
        density_img = self._density_only(frame_bgr, points, frame_bgr.shape[:2])
        _, buf2 = cv2.imencode('.jpg', density_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        density_b64 = base64.b64encode(buf2).decode('utf-8')
        r = quick_risk(count, 800, anomaly_score)
        return {
            'crowd_count':   count,
            'anomaly_score': round(anomaly_score, 4),
            'risk_score':    r['risk_score'],
            'risk_level':    r['risk_level'],
            'density_pct':   r['density_pct'],
            'alerts':        r['alerts'],
            'num_points':    len(points),
            'heatmap_b64':   heatmap_b64,
            'density_b64':   density_b64,
        }

    def _annotate_frame(self, frame, count, anomaly_score):
        out = frame.copy()
        h, w = out.shape[:2]
        color = (0,0,255) if anomaly_score>.65 else (0,165,255) if anomaly_score>.35 else (0,255,0)
        overlay = out.copy()
        cv2.rectangle(overlay, (0,0), (w,55), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        cv2.putText(out, f"COUNT: {count}",           (12,22), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
        cv2.putText(out, f"ANOMALY: {anomaly_score:.3f}", (12,44), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        bar_w = int((w-200)*min(anomaly_score,1.0))
        cv2.rectangle(out, (200,10), (200+bar_w,26), color, -1)
        cv2.rectangle(out, (200,10), (w-10,26), (80,80,80), 1)
        return out

    def _density_only(self, frame_bgr, points, shape):
        from scipy.ndimage import gaussian_filter
        h, w = shape
        density = np.zeros((h,w), dtype=np.float32)
        if points:
            for (x,y) in points:
                if 0<=x<w and 0<=y<h: density[y,x] += 1.0
        else:
            for _ in range(20):
                cx,cy = np.random.randint(50,w-50), np.random.randint(50,h-50)
                density[cy,cx] += np.random.uniform(1,5)
        density = gaussian_filter(density, sigma=25)
        if density.max()>0: density /= density.max()
        hm   = (density*255).astype(np.uint8)
        hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame_bgr, 0.5, hm_c, 0.5, 0)

    def _sim_heatmap(self, frame_bgr, count):
        from scipy.ndimage import gaussian_filter
        h,w = frame_bgr.shape[:2]
        density = np.zeros((h,w), dtype=np.float32)
        for _ in range(min(count//5,120)):
            density[np.random.randint(0,h), np.random.randint(0,w)] += 1.0
        density = gaussian_filter(density, sigma=20)
        if density.max()>0: density /= density.max()
        hm   = (density*255).astype(np.uint8)
        hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_bgr, 0.6, hm_c, 0.4, 0)
        cv2.putText(overlay, f"SIM COUNT: {count}", (10,30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
        return overlay


# ── CAMERA STREAM (FIXED) ─────────────────────────────────────
class CameraStream:
    def __init__(self, cam_id, source, zone_id, model_mgr, risk_calc):
        self.cam_id         = cam_id
        self.source         = source
        self.zone_id        = zone_id
        self.model_mgr      = model_mgr
        self.risk_calc      = risk_calc
        self.running        = False
        self.thread         = None
        self.latest_heatmap = None
        self.latest_frame   = None
        self.latest_result  = {}
        self.frame_count    = 0
        self._lock          = threading.Lock()
        # FIX: ring buffer for smooth MJPEG — keep last encoded JPEG
        self._latest_jpeg   = None

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"📷 Camera {self.cam_id} → Zone {self.zone_id} | source: {self.source}")

    def stop(self):
        self.running = False

    def _run(self):
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        # FIX: set smaller buffer to reduce latency for RTSP/video
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"⚠️  Cannot open '{self.source}' — falling back to synthetic mode")
            self._run_synthetic()
            return

        # FIX: get native FPS so we know the real delay
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / native_fps
        process_every = max(1, int(native_fps / 10))  # process ~10 fps regardless of source fps

        print(f"✅ Opened: {self.source}  ({native_fps:.1f} fps, processing every {process_every} frames)")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Loop video files from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            self.frame_count += 1

            # FIX: always encode the raw frame so MJPEG stream is smooth
            # (even if we don't run ML on every frame, we still send video)
            if self.frame_count % process_every == 0:
                # Heavy ML processing on this frame
                self._process(frame)
            else:
                # Just encode raw frame for smooth playback
                display = self.latest_heatmap if self.latest_heatmap is not None else frame
                _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            time.sleep(frame_delay)

        cap.release()

    def _run_synthetic(self):
        while self.running:
            frame = self._make_synthetic_frame()
            with self._lock:
                self.latest_frame = frame.copy()
            self._process(frame)
            time.sleep(0.1)   # FIX: faster synthetic refresh (was 2s)

    def _make_synthetic_frame(self):
        h, w = 480, 640
        frame = np.full((h,w,3), (30,30,30), dtype=np.uint8)
        cv2.rectangle(frame, (0,h//2), (w,h), (50,45,40), -1)
        for _ in range(np.random.randint(30,120)):
            cx = np.random.randint(20,w-20)
            cy = np.random.randint(h//2,h-10)
            r  = np.random.randint(5,12)
            col = (np.random.randint(100,200), np.random.randint(80,160), np.random.randint(60,140))
            cv2.circle(frame, (cx,cy), r, col, -1)
        cv2.putText(frame, f"SYNTHETIC — {self.cam_id}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        return frame

    def _process(self, frame):
        try:
            count, points, heatmap = self.model_mgr.predict_crowd(frame)
            anomaly_score          = self.model_mgr.predict_anomaly(frame)
            zone_risk              = self.risk_calc.process_zone(
                                        self.zone_id, count, anomaly_score, frame)
            annotated = self.model_mgr._annotate_frame(heatmap, count, anomaly_score)

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self.latest_heatmap = annotated
                self._latest_jpeg   = buf.tobytes()

            result = {
                'cam_id':        self.cam_id,
                'zone_id':       self.zone_id,
                'crowd_count':   count,
                'anomaly_score': round(anomaly_score, 3),
                'risk_score':    round(zone_risk.risk_score, 3),
                'risk_level':    zone_risk.risk_level.value,
                'alerts':        [a.value for a in zone_risk.alerts],
                'timestamp':     time.time()
            }
            self.latest_result = result
            socketio.emit('zone_update', result)

            if zone_risk.risk_level in (RiskLevel.RED, RiskLevel.ORANGE):
                socketio.emit('alert', {
                    'zone_id':   self.zone_id,
                    'zone_name': zone_risk.zone_name,
                    'level':     zone_risk.risk_level.value,
                    'score':     round(zone_risk.risk_score, 3),
                    'message':   f"{zone_risk.risk_level.value} in {zone_risk.zone_name} — Count: {count}"
                })
        except Exception as e:
            import traceback
            print(f"Frame error ({self.cam_id}): {e}")
            traceback.print_exc()

    def get_jpeg(self):
        """Return latest JPEG bytes — always has something to show."""
        with self._lock:
            jpg = self._latest_jpeg
        if jpg:
            return jpg
        # Placeholder
        ph = np.zeros((240,320,3), dtype=np.uint8)
        cv2.putText(ph, "Waiting for feed...", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)
        _, buf = cv2.imencode('.jpg', ph)
        return buf.tobytes()


# ── INIT ──────────────────────────────────────────────────────
model_mgr   = ModelManager()
risk_calc   = StampedeRiskCalculator(ZONE_CONFIGS)
alert_mgr   = AlertManager(cooldown_seconds=30)
venue_graph = create_stadium_graph()
cameras     = {}
floor_plans = {}


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ─── IMAGE ANALYSIS ───────────────────────────────────────────
@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400
    file     = request.files['image']
    capacity = int(request.form.get('capacity', 800))
    img_bytes = file.read()
    img_arr   = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400
    max_dim = 1280
    h, w = frame.shape[:2]
    if max(h,w) > max_dim:
        scale = max_dim / max(h,w)
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    result = model_mgr.analyze_image_full(frame)
    result['capacity'] = capacity
    r = quick_risk(result['crowd_count'], capacity, result['anomaly_score'])
    result.update({'risk_score':r['risk_score'],'risk_level':r['risk_level'],
                   'density_pct':r['density_pct'],'alerts':r['alerts']})
    return jsonify(result)


# ─── VIDEO ANALYSIS (FIXED) ───────────────────────────────────
@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """
    FIX: Sequential frame reading (no CAP_PROP_POS_MSEC seeking which is slow).
    FIX: Anomaly only run on the full clip once, not per-frame.
    FIX: Resize frames to 640px max before ML to reduce compute.
    FIX: Returns streaming JSON-lines so frontend gets progress updates.
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video field in request'}), 400

    file  = request.files['video']
    fname = secure_filename(file.filename or 'upload.mp4')
    fpath = os.path.join(UPLOAD_FOLDER, f"tmp_{int(time.time())}_{fname}")
    file.save(fpath)

    def generate():
        try:
            cap = cv2.VideoCapture(fpath)
            if not cap.isOpened():
                yield json.dumps({'error': 'Cannot open video file'}) + '\n'
                return

            fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total / fps

            # FIX: decide how many samples we want (max 60), then compute step
            num_samples = min(60, max(10, int(duration)))
            step        = max(1, int(total / num_samples))

            yield json.dumps({'progress': True, 'msg': f'Reading {num_samples} samples from {duration:.1f}s video…'}) + '\n'

            frames_all  = []
            frame_times = []
            fc          = 0

            # FIX: sequential read — skip frames by reading+discarding, not seeking
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fc % step == 0 and len(frames_all) < num_samples:
                    # FIX: resize to max 640px wide for speed
                    h, w = frame.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        frame = cv2.resize(frame, (640, int(h*scale)))
                    frames_all.append(frame)
                    frame_times.append(round(fc / fps, 1))
                fc += 1

            cap.release()

            if not frames_all:
                yield json.dumps({'error': 'No frames extracted'}) + '\n'
                return

            yield json.dumps({'progress': True, 'msg': f'Running crowd count on {len(frames_all)} frames…'}) + '\n'

            # Per-frame crowd count only (fast)
            counts    = []
            timeline  = []

            for i, frame in enumerate(frames_all):
                count, points, heatmap = model_mgr.predict_crowd(frame)
                counts.append(count)
                annotated = model_mgr._annotate_frame(heatmap, count, 0.0)
                _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
                timeline.append({
                    'time_sec':    frame_times[i],
                    'count':       count,
                    'anomaly':     0.0,   # placeholder until clip anomaly below
                    'heatmap_b64': base64.b64encode(buf).decode()
                })

            yield json.dumps({'progress': True, 'msg': 'Running ViT clip anomaly…'}) + '\n'

            # FIX: ONE clip-level anomaly call instead of N per-frame calls
            anomaly_clip = model_mgr.predict_anomaly_clip(frames_all)

            # Spread clip anomaly back onto timeline rows (good enough for display)
            for row in timeline:
                row['anomaly'] = round(anomaly_clip, 3)

            if os.path.exists(fpath):
                os.remove(fpath)

            summary = {
                'done':                  True,
                'total_frames_analyzed': len(frames_all),
                'duration_sec':          round(duration, 1),
                'clip_anomaly_score':    round(anomaly_clip, 4),
                'avg_crowd_count':       round(float(np.mean(counts)), 1),
                'max_crowd_count':       int(np.max(counts)),
                'avg_anomaly_score':     round(anomaly_clip, 4),
                'max_anomaly_score':     round(anomaly_clip, 4),
                'timeline':              timeline,
                'risk': quick_risk(int(np.max(counts)), 800, anomaly_clip)
            }
            yield json.dumps(summary) + '\n'

        except Exception as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            import traceback
            traceback.print_exc()
            yield json.dumps({'error': str(e)}) + '\n'

    return Response(stream_with_context(generate()),
                    mimetype='application/x-ndjson',
                    headers={'X-Accel-Buffering': 'no'})   # disable nginx buffering


# ─── VENUE MAP ────────────────────────────────────────────────
@app.route('/api/venue/map')
def venue_map():
    zone_risks = {}
    for cam in cameras.values():
        if cam.latest_result:
            r = cam.latest_result
            zone_risks[r['zone_id']] = r
    for zc_cfg in ZONE_CONFIGS:
        if zc_cfg['id'] not in zone_risks:
            zone_risks[zc_cfg['id']] = {
                'zone_id': zc_cfg['id'], 'crowd_count': 0,
                'risk_score': 0.05, 'risk_level': 'GREEN', 'anomaly_score': 0.0
            }
    risk_colors = {'GREEN':'#22c55e','YELLOW':'#eab308','ORANGE':'#f97316','RED':'#ef4444'}
    svg = _build_venue_svg(zone_risks, risk_colors)
    return Response(svg, mimetype='image/svg+xml')


def _build_venue_svg(zone_risks, risk_colors):
    def zc(zid):
        return risk_colors.get(zone_risks.get(zid,{}).get('risk_level','GREEN'), '#22c55e')
    def zn(zid):
        return zone_risks.get(zid,{}).get('crowd_count', 0)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600"
              style="background:#0a0e17;font-family:monospace">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect x="50" y="30" width="700" height="540" rx="20" fill="none" stroke="#1f2937" stroke-width="2"/>
  <rect x="200" y="150" width="400" height="300" rx="8"
        fill="rgba(34,197,94,0.08)" stroke="#22c55e" stroke-width="2"/>
  <text x="400" y="305" text-anchor="middle" fill="#64748b" font-size="11">MAIN STAGE</text>

  <rect x="200" y="50" width="120" height="90" rx="6"
        fill="rgba{_hex_to_rgba(zc(1),0.25)}" stroke="{zc(1)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="260" y="92" text-anchor="middle" fill="white" font-size="9">N-Stand A</text>
  <text x="260" y="105" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(1)} ppl</text>

  <rect x="340" y="50" width="120" height="90" rx="6"
        fill="rgba{_hex_to_rgba(zc(2),0.25)}" stroke="{zc(2)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="400" y="92" text-anchor="middle" fill="white" font-size="9">N-Stand B</text>
  <text x="400" y="105" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(2)} ppl</text>

  <rect x="480" y="50" width="120" height="90" rx="6"
        fill="rgba{_hex_to_rgba(zc(3),0.25)}" stroke="{zc(3)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="540" y="92" text-anchor="middle" fill="white" font-size="9">N-Stand C</text>
  <text x="540" y="105" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(3)} ppl</text>

  <rect x="200" y="460" width="120" height="90" rx="6"
        fill="rgba{_hex_to_rgba(zc(4),0.25)}" stroke="{zc(4)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="260" y="507" text-anchor="middle" fill="white" font-size="9">S-Stand A</text>
  <text x="260" y="520" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(4)} ppl</text>

  <rect x="340" y="460" width="120" height="90" rx="6"
        fill="rgba{_hex_to_rgba(zc(5),0.25)}" stroke="{zc(5)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="400" y="507" text-anchor="middle" fill="white" font-size="9">S-Stand B</text>
  <text x="400" y="520" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(5)} ppl</text>

  <rect x="480" y="460" width="120" height="90" rx="6"
        fill="rgba{_hex_to_rgba(zc(6),0.25)}" stroke="{zc(6)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="540" y="507" text-anchor="middle" fill="white" font-size="9">S-Stand C</text>
  <text x="540" y="520" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(6)} ppl</text>

  <rect x="650" y="180" width="100" height="240" rx="6"
        fill="rgba{_hex_to_rgba(zc(7),0.25)}" stroke="{zc(7)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="700" y="298" text-anchor="middle" fill="white" font-size="9">East Stand</text>
  <text x="700" y="312" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(7)} ppl</text>

  <rect x="50" y="180" width="100" height="240" rx="6"
        fill="rgba{_hex_to_rgba(zc(8),0.25)}" stroke="{zc(8)}" stroke-width="1.5" filter="url(#glow)"/>
  <text x="100" y="298" text-anchor="middle" fill="white" font-size="9">West Stand</text>
  <text x="100" y="312" text-anchor="middle" fill="#94a3b8" font-size="8">{zn(8)} ppl</text>

  <rect x="200" y="148" width="400" height="15" rx="3"
        fill="rgba{_hex_to_rgba(zc(9),0.4)}" stroke="{zc(9)}" stroke-width="1"/>
  <text x="400" y="160" text-anchor="middle" fill="{zc(9)}" font-size="8">N.CONCOURSE ({zn(9)})</text>

  <rect x="200" y="437" width="400" height="15" rx="3"
        fill="rgba{_hex_to_rgba(zc(10),0.4)}" stroke="{zc(10)}" stroke-width="1"/>
  <text x="400" y="449" text-anchor="middle" fill="{zc(10)}" font-size="8">S.CONCOURSE ({zn(10)})</text>

  <rect x="210" y="145" width="50" height="20" rx="3"
        fill="rgba{_hex_to_rgba(zc(13),0.5)}" stroke="{zc(13)}" stroke-width="1"/>
  <text x="235" y="158" text-anchor="middle" fill="white" font-size="7">NTL</text>

  <rect x="540" y="145" width="50" height="20" rx="3"
        fill="rgba{_hex_to_rgba(zc(14),0.5)}" stroke="{zc(14)}" stroke-width="1"/>
  <text x="565" y="158" text-anchor="middle" fill="white" font-size="7">NTR</text>

  <circle cx="80"  cy="50"  r="14" fill="#1a2744" stroke="#3b82f6" stroke-width="2"/>
  <text x="80"  y="55"  text-anchor="middle" fill="#3b82f6" font-size="9" font-weight="bold">EXIT</text>
  <circle cx="720" cy="50"  r="14" fill="#1a2744" stroke="#3b82f6" stroke-width="2"/>
  <text x="720" y="55"  text-anchor="middle" fill="#3b82f6" font-size="9" font-weight="bold">EXIT</text>
  <circle cx="80"  cy="550" r="14" fill="#1a2744" stroke="#3b82f6" stroke-width="2"/>
  <text x="80"  y="555" text-anchor="middle" fill="#3b82f6" font-size="9" font-weight="bold">EXIT</text>
  <circle cx="720" cy="550" r="14" fill="#1a2744" stroke="#3b82f6" stroke-width="2"/>
  <text x="720" y="555" text-anchor="middle" fill="#3b82f6" font-size="9" font-weight="bold">EXIT</text>
  <circle cx="780" cy="300" r="14" fill="#1a2744" stroke="#3b82f6" stroke-width="2"/>
  <text x="780" y="305" text-anchor="middle" fill="#3b82f6" font-size="9" font-weight="bold">EXIT</text>
  <circle cx="20"  cy="300" r="14" fill="#1a2744" stroke="#3b82f6" stroke-width="2"/>
  <text x="20"  y="305" text-anchor="middle" fill="#3b82f6" font-size="9" font-weight="bold">EXIT</text>

  <text x="400" y="590" text-anchor="middle" fill="#334155" font-size="10">
    StampedeShield — Live Venue Map
  </text>
</svg>'''
    return svg


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r,g,b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"({r},{g},{b},{alpha})"


# ─── EXISTING ROUTES ──────────────────────────────────────────
@app.route('/api/status')
def get_status():
    zone_summary = {}
    for cam in cameras.values():
        if cam.latest_result:
            zone_summary[cam.latest_result['zone_id']] = cam.latest_result
    for zc_cfg in ZONE_CONFIGS:
        if zc_cfg['id'] not in zone_summary:
            zone_summary[zc_cfg['id']] = {
                'zone_id': zc_cfg['id'], 'crowd_count': 0,
                'anomaly_score': 0.0, 'risk_score': 0.05,
                'risk_level': 'GREEN', 'alerts': []
            }
    scores   = [z['risk_score'] for z in zone_summary.values()]
    max_risk = max(scores) if scores else 0
    status   = ('CRITICAL' if max_risk>=0.75 else 'DANGER' if max_risk>=0.55
                 else 'CAUTION' if max_risk>=0.30 else 'NORMAL')
    return jsonify({'status':status,'max_risk':round(max_risk,3),
                    'zones':zone_summary,'timestamp':time.time()})


@app.route('/api/routing/<int:zone_id>')
def get_routing(zone_id):
    try:
        for cam in cameras.values():
            if cam.latest_result:
                nid = cam.latest_result['zone_id']
                if nid in venue_graph.nodes:
                    venue_graph.nodes[nid]['risk']          = cam.latest_result['risk_score']
                    venue_graph.nodes[nid]['current_crowd'] = cam.latest_result['crowd_count']
        update_edge_weights(venue_graph)
        routes = find_all_safe_exits(venue_graph, zone_id)
        return jsonify({'zone_id':zone_id,'routes':routes[:3],'best':routes[0] if routes else None})
    except Exception as e:
        return jsonify({'error':str(e)}), 500


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    return jsonify({
        cid: {'zone_id':c.zone_id,'running':c.running,
              'source':c.source,'latest':c.latest_result}
        for cid,c in cameras.items()
    })


@app.route('/api/cameras/add', methods=['POST'])
def add_camera():
    data    = request.json
    cam_id  = data.get('cam_id', f'CAM{len(cameras)+1:02d}')
    source  = data.get('source')
    zone_id = int(data.get('zone_id', 1))
    if not source:
        return jsonify({'error':'source required'}), 400
    if cam_id in cameras:
        cameras[cam_id].stop()
    cam = CameraStream(cam_id, source, zone_id, model_mgr, risk_calc)
    cam.start()
    cameras[cam_id] = cam
    return jsonify({'success':True,'cam_id':cam_id,'zone_id':zone_id})


@app.route('/api/cameras/<cam_id>/stop', methods=['POST'])
def stop_camera(cam_id):
    if cam_id in cameras:
        cameras[cam_id].stop()
        return jsonify({'success':True})
    return jsonify({'error':'Not found'}), 404


@app.route('/api/cameras/<cam_id>/stream')
def camera_stream(cam_id):
    """
    FIX: proper MJPEG stream — no sleep between frames, boundary correct.
    Frontend img tag refreshes itself via JS loop (see index.html fix).
    """
    def generate():
        # FIX: target ~25fps delivery to browser
        interval = 1.0 / 25
        while True:
            cam = cameras.get(cam_id)
            jpg = cam.get_jpeg() if cam else None
            if jpg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n'
                       b'\r\n' + jpg + b'\r\n')
            time.sleep(interval)

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control':    'no-cache, no-store, must-revalidate',
            'Pragma':           'no-cache',
            'Expires':          '0',
            'X-Accel-Buffering':'no',   # nginx: don't buffer MJPEG
        }
    )


@app.route('/api/floorplan/upload', methods=['POST'])
def upload_floorplan():
    if 'file' not in request.files:
        return jsonify({'error':'No file'}), 400
    file  = request.files['file']
    venue = request.form.get('venue_name','venue')
    img_bytes = file.read()
    img_arr   = np.frombuffer(img_bytes, np.uint8)
    img       = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error':'Invalid image'}), 400
    img   = cv2.resize(img, (1200,800))
    fname = f"{venue}_{int(time.time())}.jpg"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    cv2.imwrite(fpath, img)
    _,buf = cv2.imencode('.jpg', img)
    b64   = base64.b64encode(buf).decode('utf-8')
    floor_plans[venue] = {'path':fpath,'zones':[],'width':1200,'height':800}
    return jsonify({'success':True,'venue':venue,'image_b64':b64,'width':1200,'height':800})


@app.route('/api/floorplan/zones', methods=['POST'])
def save_zones():
    data  = request.json
    venue = data.get('venue')
    zones = data.get('zones',[])
    if venue not in floor_plans:
        return jsonify({'error':'Not found'}), 404
    floor_plans[venue]['zones'] = zones
    img = cv2.imread(floor_plans[venue]['path'])
    for z in zones:
        x,y,w,h = int(z['x']),int(z['y']),int(z['w']),int(z['h'])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,z['name'],(x+5,y+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    _,buf = cv2.imencode('.jpg',img)
    return jsonify({'success':True,'zones_saved':len(zones),
                    'preview_b64':base64.b64encode(buf).decode()})


@app.route('/api/demo/start', methods=['POST'])
def start_demo():
    def simulate():
        t = 0
        base = {1:(600,0.05),2:(580,0.06),3:(550,0.04),
                4:(620,0.05),7:(900,0.08),8:(850,0.07)}
        while True:
            t += 1
            for zid,(bc,ba) in base.items():
                crowd   = bc + np.random.randint(-50,50)
                anomaly = ba + np.random.uniform(-0.02,0.02)
                if t>30 and zid in (1,2,9):
                    crowd   = min(crowd+t*5, 795)
                    anomaly = min(anomaly+t*0.02, 0.95)
                r = quick_risk(int(crowd), 800, float(anomaly))
                socketio.emit('zone_update',{
                    'zone_id':zid,'crowd_count':int(crowd),
                    'anomaly_score':round(float(anomaly),3),
                    'risk_score':r['risk_score'],'risk_level':r['risk_level'],
                    'alerts':r['alerts'],'timestamp':time.time()
                })
            time.sleep(2)
    threading.Thread(target=simulate, daemon=True).start()
    return jsonify({'success':True})


# ── WEBSOCKET ─────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('connected', {'status':'ok'})

@socketio.on('request_routing')
def on_routing(data):
    zone_id = data.get('zone_id',1)
    try:
        update_edge_weights(venue_graph)
        routes = find_all_safe_exits(venue_graph, zone_id)
        emit('routing_response',{
            'zone_id':zone_id,'routes':routes[:3],
            'best':routes[0] if routes else None
        })
    except Exception as e:
        emit('routing_response',{'error':str(e)})


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  STAMPEDESHIELD — CROWD SAFETY SYSTEM")
    print("  http://localhost:5000")
    print("="*55)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)