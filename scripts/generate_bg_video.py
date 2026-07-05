"""
Generate the landing page background video (network-bg.mp4).
Run once: python scripts/generate_bg_video.py

Output: assets/network-bg.mp4  (~3 MB, 10-second loop at 30fps, 1280×720)
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFilter

# ── Config ────────────────────────────────────────────────────────────
W, H    = 1280, 720
FPS     = 30
SECONDS = 10
FRAMES  = FPS * SECONDS

N_NODES   = 60
HUB_FRAC  = 0.12          # fraction that are larger "hub" nodes
SPEED     = 0.4
MAX_DIST  = 210
HUB_EXTRA = 1.5           # hubs connect further

BG_TOP    = (10,  15, 30)
BG_BOT    = (13,  21, 48)
NODE_COL  = (91, 141, 239)   # #5B8DEF blue
HUB_COL   = (76, 175, 125)   # #4CAF7D green
EDGE_COL  = (91, 141, 239)

OUT_PATH  = Path(__file__).parent.parent / "assets" / "network-bg.mp4"
# ──────────────────────────────────────────────────────────────────────


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def alpha_blend(base: np.ndarray, color: tuple, alpha: float, x0, y0, x1, y1):
    """Blend a solid colour rectangle into the frame array with alpha."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x0 >= x1 or y0 >= y1:
        return
    patch = base[y0:y1, x0:x1].astype(float)
    for c in range(3):
        patch[:, :, c] = patch[:, :, c] * (1 - alpha) + color[c] * alpha
    base[y0:y1, x0:x1] = patch.astype(np.uint8)


def draw_glow_node(frame: np.ndarray, x, y, r, col, intensity=0.22):
    """Simple radial glow: concentric filled circles decreasing in alpha."""
    for step in range(5, 0, -1):
        gr = r * (1 + step * 1.1)
        a  = intensity * (1 - step / 6.0)
        ix0, iy0 = int(x - gr), int(y - gr)
        ix1, iy1 = int(x + gr), int(y + gr)
        alpha_blend(frame, col, a, ix0, iy0, ix1, iy1)


def draw_circle(frame: np.ndarray, cx, cy, r, col):
    """Fill a hard circle into the frame array."""
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(W, int(cx + r) + 1), min(H, int(cy + r) + 1)
    for py in range(y0, y1):
        for px in range(x0, x1):
            if (px - cx)**2 + (py - cy)**2 <= r**2:
                frame[py, px] = col


def make_bg_gradient():
    """Pre-render the background gradient as a numpy array."""
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        col = lerp_color(BG_TOP, BG_BOT, t)
        bg[y, :] = col
    return bg


def spawn_nodes():
    n_hubs = int(N_NODES * HUB_FRAC)
    nodes = []
    for i in range(N_NODES):
        hub = i < n_hubs
        angle = random.uniform(0, 2 * math.pi)
        speed = SPEED * (0.45 if hub else 1.0) * random.uniform(0.6, 1.0)
        nodes.append({
            "x":     random.uniform(0, W),
            "y":     random.uniform(0, H),
            "vx":    math.cos(angle) * speed,
            "vy":    math.sin(angle) * speed,
            "r":     random.uniform(4.5, 7.5) if hub else random.uniform(1.5, 3.5),
            "hub":   hub,
            "phase": random.uniform(0, 2 * math.pi),
            "col":   HUB_COL if hub else NODE_COL,
        })
    return nodes


def step_nodes(nodes):
    for n in nodes:
        n["x"] += n["vx"]
        n["y"] += n["vy"]
        if n["x"] < -60:  n["x"] = W + 60
        if n["x"] > W+60: n["x"] = -60
        if n["y"] < -60:  n["y"] = H + 60
        if n["y"] > H+60: n["y"] = -60


def render_frame(bg, nodes, t):
    frame = bg.copy()

    # --- Edges (draw into a PIL image for anti-aliased lines, then composite) ---
    edge_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_img)

    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            dx, dy = a["x"] - b["x"], a["y"] - b["y"]
            d = math.sqrt(dx*dx + dy*dy)
            thresh = MAX_DIST * (HUB_EXTRA if (a["hub"] or b["hub"]) else 1.0)
            if d < thresh:
                alpha_f = (1 - d / thresh) ** 1.8 * 0.32
                a_val = int(alpha_f * 255)
                r, g, b_col = EDGE_COL
                edge_draw.line(
                    [(int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"]))],
                    fill=(r, g, b_col, a_val),
                    width=1,
                )

    # Composite edges onto frame
    bg_pil = Image.fromarray(frame, "RGB").convert("RGBA")
    bg_pil = Image.alpha_composite(bg_pil, edge_img).convert("RGB")
    frame = np.array(bg_pil)

    # --- Nodes (glow + core) ---
    for n in nodes:
        pulse = 1 + math.sin(t * 1.1 + n["phase"]) * 0.2 if n["hub"] else 1.0
        r = n["r"] * pulse
        draw_glow_node(frame, n["x"], n["y"], r, n["col"], intensity=0.18)
        draw_circle(frame, n["x"], n["y"], r, n["col"])

    return frame


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {FRAMES} frames ({SECONDS}s @ {FPS}fps) → {OUT_PATH}")

    bg = make_bg_gradient()
    nodes = spawn_nodes()

    writer = imageio.get_writer(
        str(OUT_PATH),
        fps=FPS,
        codec="libx264",
        quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p"],  # broad browser compatibility
    )

    for frame_idx in range(FRAMES):
        t = frame_idx / FPS
        step_nodes(nodes)
        frame = render_frame(bg, nodes, t)
        writer.append_data(frame)

        if frame_idx % 30 == 0:
            pct = frame_idx / FRAMES * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)

    writer.close()
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nDone — {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
