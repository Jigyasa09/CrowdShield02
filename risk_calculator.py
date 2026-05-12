# risk_calculator.py
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque


class RiskLevel(Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED    = "RED"

class AlertType(Enum):
    DENSITY_HIGH    = "High crowd density detected"
    ANOMALY         = "Anomaly behavior detected"
    TURBULENCE      = "Panic movement detected"
    COUNTER_FLOW    = "Counter-flow detected"
    CAPACITY_BREACH = "Zone capacity exceeded"
    STAMPEDE_RISK   = "STAMPEDE RISK — EVACUATE IMMEDIATELY"

WEIGHTS = {'density': 0.35, 'anomaly': 0.40, 'turbulence': 0.25}

THRESHOLDS = {
    RiskLevel.GREEN:  (0.00, 0.30),
    RiskLevel.YELLOW: (0.30, 0.55),
    RiskLevel.ORANGE: (0.55, 0.75),
    RiskLevel.RED:    (0.75, 1.00),
}


@dataclass
class ZoneMetrics:
    zone_id:       int
    zone_name:     str
    capacity:      int
    crowd_count:   int   = 0
    anomaly_score: float = 0.0
    turbulence:    float = 0.0
    counter_flow:  float = 0.0
    timestamp:     float = field(default_factory=time.time)

    @property
    def density_ratio(self):
        return min(self.crowd_count / max(self.capacity, 1), 1.0)

    @property
    def density_score(self):
        d = self.density_ratio
        if d < 0.5:    return d * 0.4
        elif d < 0.75: return 0.2 + (d - 0.5) * 1.2
        else:          return 0.5 + (d - 0.75) * 2.0


@dataclass
class ZoneRisk:
    zone_id:    int
    zone_name:  str
    risk_score: float
    risk_level: RiskLevel
    alerts:     List[AlertType]
    metrics:    ZoneMetrics
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self):
        return {
            'zone_id':       self.zone_id,
            'zone_name':     self.zone_name,
            'risk_score':    round(self.risk_score, 3),
            'risk_level':    self.risk_level.value,
            'alerts':        [a.value for a in self.alerts],
            'crowd_count':   self.metrics.crowd_count,
            'capacity':      self.metrics.capacity,
            'density_pct':   round(self.metrics.density_ratio * 100, 1),
            'anomaly_score': round(self.metrics.anomaly_score, 3),
            'turbulence':    round(self.metrics.turbulence, 3),
            'timestamp':     self.timestamp
        }


class FlowAnalyzer:
    def __init__(self, history_len=5):
        self.prev_frame    = None
        self.flow_history  = deque(maxlen=history_len)

    def analyze(self, frame: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_frame = gray
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.flow_history.append((magnitude, angle))

        turbulence = float(np.clip(
            np.std(angle) / 2.0 * 0.6 + min(np.mean(magnitude) / 10.0, 1.0) * 0.4,
            0, 1))

        hist, _ = np.histogram(angle.flatten(), bins=8, range=(0, 2 * np.pi))
        hist_norm  = hist / (hist.sum() + 1e-8)
        entropy    = -np.sum(hist_norm * np.log(hist_norm + 1e-8))
        counter_flow = float(np.clip(entropy / 2.0, 0, 1))

        return turbulence, counter_flow


class StampedeRiskCalculator:
    def __init__(self, zone_configs: List[Dict]):
        self.zones          = {z['id']: z for z in zone_configs}
        self.flow_analyzers = {z['id']: FlowAnalyzer() for z in zone_configs}
        self.alert_history  = deque(maxlen=100)
        self.risk_history   = {z['id']: deque(maxlen=30) for z in zone_configs}

    def compute_zone_risk(self, metrics: ZoneMetrics) -> ZoneRisk:
        risk_score = (
            WEIGHTS['density']    * metrics.density_score  +
            WEIGHTS['anomaly']    * metrics.anomaly_score  +
            WEIGHTS['turbulence'] * metrics.turbulence
        )
        risk_score = float(np.clip(risk_score, 0, 1))

        # Spike amplification
        history = self.risk_history[metrics.zone_id]
        if len(history) >= 3:
            spike = risk_score - np.mean(list(history)[-3:])
            if spike > 0.2:
                risk_score = min(risk_score * 1.15, 1.0)
        self.risk_history[metrics.zone_id].append(risk_score)

        risk_level = RiskLevel.GREEN
        for level, (lo, hi) in THRESHOLDS.items():
            if lo <= risk_score < hi:
                risk_level = level; break
        if risk_score >= 0.75:
            risk_level = RiskLevel.RED

        alerts = []
        if metrics.density_ratio >= 0.90: alerts.append(AlertType.CAPACITY_BREACH)
        elif metrics.density_ratio >= 0.75: alerts.append(AlertType.DENSITY_HIGH)
        if metrics.anomaly_score  >= 0.65: alerts.append(AlertType.ANOMALY)
        if metrics.turbulence     >= 0.60: alerts.append(AlertType.TURBULENCE)
        if metrics.counter_flow   >= 0.65: alerts.append(AlertType.COUNTER_FLOW)
        if risk_score             >= 0.75: alerts.append(AlertType.STAMPEDE_RISK)

        return ZoneRisk(
            zone_id    = metrics.zone_id,
            zone_name  = metrics.zone_name,
            risk_score = risk_score,
            risk_level = risk_level,
            alerts     = alerts,
            metrics    = metrics
        )

    def process_zone(self, zone_id, crowd_count, anomaly_score,
                     frame=None) -> ZoneRisk:
        zone = self.zones[zone_id]
        turbulence = counter_flow = 0.0
        if frame is not None:
            turbulence, counter_flow = \
                self.flow_analyzers[zone_id].analyze(frame)

        metrics = ZoneMetrics(
            zone_id       = zone_id,
            zone_name     = zone['name'],
            capacity      = zone['capacity'],
            crowd_count   = crowd_count,
            anomaly_score = float(np.clip(anomaly_score, 0, 1)),
            turbulence    = turbulence,
            counter_flow  = counter_flow
        )
        return self.compute_zone_risk(metrics)

    def process_all_zones(self, zone_data: Dict) -> Dict:
        return {
            zid: self.process_zone(
                zone_id       = zid,
                crowd_count   = data.get('crowd_count', 0),
                anomaly_score = data.get('anomaly_score', 0.0),
                frame         = data.get('frame')
            )
            for zid, data in zone_data.items()
        }

    def get_system_status(self, zone_risks: Dict) -> Dict:
        if not zone_risks:
            return {'status': 'UNKNOWN', 'max_risk': 0}
        scores   = [r.risk_score for r in zone_risks.values()]
        max_risk = max(scores)
        red      = [r for r in zone_risks.values() if r.risk_level == RiskLevel.RED]
        orange   = [r for r in zone_risks.values() if r.risk_level == RiskLevel.ORANGE]
        critical = max(zone_risks.values(), key=lambda r: r.risk_score)

        if max_risk >= 0.75 or red:       status = 'CRITICAL'
        elif max_risk >= 0.55 or len(orange) >= 2: status = 'DANGER'
        elif max_risk >= 0.30:             status = 'CAUTION'
        else:                              status = 'NORMAL'

        return {
            'status':           status,
            'max_risk':         round(max_risk, 3),
            'avg_risk':         round(float(np.mean(scores)), 3),
            'red_zones':        len(red),
            'orange_zones':     len(orange),
            'critical_zone':    critical.zone_name,
            'critical_zone_id': critical.zone_id,
            'total_alerts':     sum(len(r.alerts) for r in zone_risks.values()),
            'zone_summary':     {zid: r.to_dict() for zid, r in zone_risks.items()}
        }


class AlertManager:
    def __init__(self, cooldown_seconds=30):
        self.cooldown   = cooldown_seconds
        self.last_alert = {}

    def should_alert(self, zone_id, risk_level):
        if risk_level in (RiskLevel.GREEN, RiskLevel.YELLOW):
            return False
        key  = f"{zone_id}_{risk_level.value}"
        now  = time.time()
        cd   = 10 if risk_level == RiskLevel.RED else self.cooldown
        if now - self.last_alert.get(key, 0) > cd:
            self.last_alert[key] = now
            return True
        return False

    def format_message(self, zone_risk: ZoneRisk) -> str:
        emoji = {'GREEN':'✅','YELLOW':'⚠️','ORANGE':'🟠','RED':'🚨'}
        e = emoji.get(zone_risk.risk_level.value, '⚠️')
        return (f"{e} {zone_risk.risk_level.value} — {zone_risk.zone_name} | "
                f"Risk:{zone_risk.risk_score:.2f} | "
                f"Crowd:{zone_risk.metrics.crowd_count}/{zone_risk.metrics.capacity}")


def quick_risk(crowd_count, capacity, anomaly_score, turbulence=0.0):
    """Quick risk calculation without full pipeline — useful for testing."""
    d = min(crowd_count / max(capacity, 1), 1.0)
    if d < 0.5:    ds = d * 0.4
    elif d < 0.75: ds = 0.2 + (d-0.5)*1.2
    else:          ds = 0.5 + (d-0.75)*2.0

    risk = float(np.clip(
        WEIGHTS['density']*ds + WEIGHTS['anomaly']*anomaly_score +
        WEIGHTS['turbulence']*turbulence, 0, 1))

    level = RiskLevel.GREEN
    for lv, (lo, hi) in THRESHOLDS.items():
        if lo <= risk < hi: level = lv; break
    if risk >= 0.75: level = RiskLevel.RED

    alerts = []
    if d    >= 0.90: alerts.append("Capacity breach")
    if d    >= 0.75: alerts.append("High density")
    if anomaly_score >= 0.65: alerts.append("Anomaly detected")
    if turbulence    >= 0.60: alerts.append("Panic movement")
    if risk >= 0.75:          alerts.append("STAMPEDE RISK")

    return {'risk_score': round(risk,3), 'risk_level': level.value,
            'density_pct': round(d*100,1), 'alerts': alerts}