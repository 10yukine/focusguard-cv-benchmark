"""FocusGuard: phone-detection attention monitor that doubles as a detector benchmark.

Live mode   : python focusguard.py                 ([T] switch model, [Q] quit)
Benchmark   : python focusguard.py --benchmark 50  (prints a latency table for all models)
"""

import argparse
import os
import time
from typing import List, Tuple

import cv2
import torch
from torchvision.models import detection as tvd
from torchvision.transforms import functional as F
from ultralytics import YOLO

MODES = ("yolo", "ssd", "rcnn")
NAMES = {"yolo": "YOLOv8n", "ssd": "SSD300-VGG16", "rcnn": "Faster R-CNN R50-FPN v2"}
CONF = {"yolo": 0.25, "ssd": 0.25, "rcnn": 0.80}
# Live mode runs the slow models on every Nth frame and reuses the boxes in between,
# so the preview stays smooth. The benchmark ignores this and times every frame.
STRIDE = {"yolo": 1, "ssd": 5, "rcnn": 10}
PHONE = "cell phone"
ALERT_HOLD_FRAMES = 10  # keep the alert up briefly so one missed detection doesn't flicker it
MAIN_WIN, ALERT_WIN = "FocusGuard Benchmark", "FOCUS ALERT"

Detection = Tuple[Tuple[int, int, int, int], str, float]  # (x1, y1, x2, y2), class name, score


class Detectors:
    """Loads each model on first use and gives all three the same detect() interface."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = {}

    def _get(self, mode: str):
        if mode not in self._models:
            print(f"[FocusGuard] Loading {NAMES[mode]} on {self.device.upper()}...")
            if mode == "yolo":
                self._models[mode] = (YOLO("yolov8n.pt"), None)
            else:
                build, weights = {
                    "ssd": (tvd.ssd300_vgg16, tvd.SSD300_VGG16_Weights.DEFAULT),
                    "rcnn": (tvd.fasterrcnn_resnet50_fpn_v2, tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
                }[mode]
                model = build(weights=weights).to(self.device).eval()
                self._models[mode] = (model, weights.meta["categories"])
        return self._models[mode]

    @torch.inference_mode()
    def detect(self, mode: str, frame_bgr) -> List[Detection]:
        model, categories = self._get(mode)
        if mode == "yolo":
            res = model(frame_bgr, verbose=False, conf=CONF[mode], device=self.device)[0]
            boxes, labels, scores, names = res.boxes.xyxy, res.boxes.cls, res.boxes.conf, res.names
        else:
            tensor = F.to_tensor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).to(self.device)
            pred = model([tensor])[0]
            boxes, labels, scores, names = pred["boxes"], pred["labels"], pred["scores"], categories
        # .tolist() copies to CPU, which also waits for the GPU to finish, so timings are honest.
        return [
            (tuple(int(v) for v in box), names[int(label)], score)
            for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist())
            if score >= CONF[mode]
        ]


def open_source(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source '{source}'.")
    return cap


def draw(frame, detections: List[Detection]) -> None:
    for (x1, y1, x2, y2), name, score in detections:
        color = (0, 0, 255) if name == PHONE else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{name} {score:.2f}", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )


def show_alert(vid: cv2.VideoCapture) -> None:
    vid.grab()  # skip a frame so the clip keeps roughly real-time pace
    ok, clip = vid.read()
    if not ok:  # end of clip: loop
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, clip = vid.read()
    if ok:
        cv2.imshow(ALERT_WIN, cv2.resize(clip, (480, int(480 * clip.shape[0] / clip.shape[1]))))


def run_live(source: str, alert_video: str) -> None:
    detectors = Detectors()
    cap = open_source(source)
    alert = cv2.VideoCapture(alert_video) if os.path.exists(alert_video) else None
    if alert is None:
        print(f"[FocusGuard] '{alert_video}' not found: alerts will be on-screen text only.")
    print("Controls: [T] cycle model (YOLO -> SSD -> Faster R-CNN) | [Q] quit")

    mode_idx = frame_no = hold = 0
    detections: List[Detection] = []
    infer_ms = 0.0
    alert_open = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        mode = MODES[mode_idx]

        if frame_no % STRIDE[mode] == 0:
            t0 = time.perf_counter()
            detections = detectors.detect(mode, frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0
        frame_no += 1

        phone_seen = any(name == PHONE for _, name, _ in detections)
        hold = ALERT_HOLD_FRAMES if phone_seen else max(0, hold - 1)

        draw(frame, detections)
        status, color = ("PHONE DETECTED", (0, 0, 255)) if hold else ("FOCUSED", (0, 255, 0))
        cv2.putText(
            frame,
            f"[{NAMES[mode]}] {status} | {infer_ms:.0f} ms/inference ({1000.0 / max(infer_ms, 1e-3):.1f} FPS)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        cv2.imshow(MAIN_WIN, frame)

        if hold and alert:
            show_alert(alert)
            alert_open = True
        elif alert_open:
            cv2.destroyWindow(ALERT_WIN)
            alert_open = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t"):
            mode_idx = (mode_idx + 1) % len(MODES)
            detections, frame_no, hold = [], 0, 0

    cap.release()
    if alert:
        alert.release()
    cv2.destroyAllWindows()


def run_benchmark(source: str, n_frames: int) -> None:
    """Time every model on the same captured frames and print a Markdown table."""
    cap = open_source(source)
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("No frames captured.")

    detectors = Detectors()
    rows = []
    for mode in MODES:
        detectors.detect(mode, frames[0])  # warm-up: model load, cuDNN autotune, lazy init
        n_dets = 0
        t0 = time.perf_counter()
        for frame in frames:
            n_dets += len(detectors.detect(mode, frame))
        mean_ms = (time.perf_counter() - t0) * 1000.0 / len(frames)
        rows.append(f"| {NAMES[mode]} | {mean_ms:.1f} | {1000.0 / mean_ms:.1f} | {n_dets / len(frames):.1f} |")

    h, w = frames[0].shape[:2]
    print(f"\n{len(frames)} frames at {w}x{h} on {detectors.device.upper()}\n")
    print("| Model | Mean latency (ms) | FPS | Avg detections/frame |")
    print("|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocusGuard Multi-Architecture Detection Benchmark")
    parser.add_argument("--source", default="0", help="Webcam index or path to a video file")
    parser.add_argument("--video", default="subway_surfers.mp4", help="Clip played while a phone is visible")
    parser.add_argument("--benchmark", type=int, metavar="N", help="Time all models on N frames, then exit")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.source, args.benchmark)
    else:
        run_live(args.source, args.video)
