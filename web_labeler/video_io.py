from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import os
import logging


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


@dataclass
class VideoMeta:
    total_frames: int
    fps: float
    width: int
    height: int


class VideoReader:
    """
    Small wrapper around cv2.VideoCapture with a HEVC-friendly seek strategy
    similar to what `labelingMvpV6_1.py` does.
    """

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[Path] = None
        self.meta: Optional[VideoMeta] = None
        self.frame_idx: int = -1
        self._last_bgr = None
        self.end_reached: bool = False
        self._debug = os.getenv("LABELER_DEBUG", "").strip() not in ("", "0", "false", "False")
        self._log = logging.getLogger("labeler.video")

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.video_path = None
        self.meta = None
        self.frame_idx = -1
        self._last_bgr = None
        self.end_reached = False

    def open(self, video_path: Path) -> VideoMeta:
        self.close()
        self.video_path = Path(video_path)

        cap = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG) if hasattr(cv2, "CAP_FFMPEG") else cv2.VideoCapture(
            str(self.video_path)
        )
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.cap = cap
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read first frame")
        h, w = frame.shape[:2]
        self._last_bgr = frame
        self.frame_idx = 0

        self.meta = VideoMeta(total_frames=total_frames, fps=fps, width=w, height=h)
        return self.meta

    def _set_pos_frames(self, idx: int):
        assert self.cap is not None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

    def _set_pos_msec(self, msec: int):
        assert self.cap is not None
        self.cap.set(cv2.CAP_PROP_POS_MSEC, int(msec))

    def safe_seek_read(self, target_idx: int) -> Tuple[int, "cv2.typing.MatLike"]:
        """
        Returns (frame_idx, frame_bgr) for the requested index.
        We try an exact frame seek first; if the codec/container is seek-hostile (e.g. some HEVC),
        we fall back to a backseek-by-time and read forward until we reach the target.
        """
        if self.cap is None or self.meta is None:
            raise RuntimeError("Video not opened")

        self.end_reached = False

        # Frame count can be wrong/unreliable for some codecs/containers.
        # If total_frames is known (>0) clamp; otherwise, accept requested index.
        if (self.meta.total_frames or 0) > 0:
            target_idx = clamp(int(target_idx), 0, max(0, (self.meta.total_frames or 1) - 1))
        else:
            target_idx = int(target_idx)

        # 0) Fast path for sequential forward reads (playback).
        # If the requested frame is slightly ahead of what we just returned,
        # reading forward is both faster and avoids keyframe/seek jitter.
        if self.frame_idx >= 0 and target_idx >= self.frame_idx:
            delta = target_idx - self.frame_idx
            if 0 < delta <= 64:
                assert self.cap is not None
                # grab delta-1 frames, then read the target
                for _ in range(max(0, delta - 1)):
                    ok = self.cap.grab()
                    if not ok:
                        break
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    idx = max(0, pos - 1)
                    self.frame_idx = idx
                    self._last_bgr = frame
                    if self._debug and (idx % 250 == 0):
                        self._log.debug("sequential: target=%s got=%s", target_idx, idx)
                    return idx, frame

        # 1) Try exact frame seek first. This avoids "jumping backward" behavior.
        try:
            self._set_pos_frames(target_idx)
            ok, frame = self.cap.read()
            if ok and frame is not None:
                pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                idx = max(0, pos - 1)
                self.frame_idx = idx
                self._last_bgr = frame
                if idx == target_idx:
                    if self._debug and (idx % 250 == 0):
                        self._log.debug("exact: target=%s got=%s", target_idx, idx)
                    return idx, frame
        except Exception:
            pass

        # 2) Frame-based backseek and forward read.
        # This is more reliable than time-based seeking when FPS metadata is wrong.
        start_idx = max(0, target_idx - 90)
        self._set_pos_frames(start_idx)

        last = None
        last_idx = -1
        max_reads = min(800, (target_idx - start_idx) + 200)
        for _ in range(max_reads):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.frame_idx = max(0, pos - 1)
            last = frame
            last_idx = self.frame_idx
            if self.frame_idx >= target_idx:
                self._last_bgr = frame
                # If we overshot, try exact seek one more time to land precisely.
                if self.frame_idx != target_idx:
                    try:
                        self._set_pos_frames(target_idx)
                        ok2, frame2 = self.cap.read()
                        if ok2 and frame2 is not None:
                            pos2 = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                            idx2 = max(0, pos2 - 1)
                            self.frame_idx = idx2
                            self._last_bgr = frame2
                            return idx2, frame2
                    except Exception:
                        pass
                return self.frame_idx, frame

        # 3) End-of-video or decode hiccup: return the last readable frame and mark end reached.
        # Try stepping backward a bit from target to find a decodable frame.
        for back in range(0, 80):
            cand = max(0, target_idx - back)
            try:
                self._set_pos_frames(cand)
                ok3, frame3 = self.cap.read()
                if ok3 and frame3 is not None:
                    pos3 = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    idx3 = max(0, pos3 - 1)
                    self.frame_idx = idx3
                    self._last_bgr = frame3
                    if idx3 < target_idx:
                        self.end_reached = True
                    if self._debug:
                        self._log.debug("fallback-back: target=%s got=%s back=%s end=%s", target_idx, idx3, back, int(self.end_reached))
                    return idx3, frame3
            except Exception:
                continue

        if last is not None and last_idx >= 0:
            self._last_bgr = last
            self.frame_idx = last_idx
            self.end_reached = True
            return last_idx, last

        raise RuntimeError(f"Could not read frame {target_idx}")


