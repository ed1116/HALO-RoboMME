"""
Helpers for rendering QA data points as self-contained HTML pages.

All assets (videos, frame images) are base64-embedded so the resulting HTML
file can be opened directly in VSCode Simple Browser without a local server.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np



def html_escape(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def draw_frame_number(frame: np.ndarray, frame_num: int) -> np.ndarray:
    import cv2
    out = frame.copy()
    text = f"Frame {frame_num}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(out, (4, 4), (4 + tw + 4, 4 + th + baseline + 4), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 4 + th + 2), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def save_video_mp4(frames_rgb: np.ndarray, out_path: str, fps: int = 10) -> None:
    import cv2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T, H, W, _ = frames_rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def video_to_b64(mp4_path: str) -> str:
    import base64
    with open(mp4_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def frame_to_b64_jpeg(frame_rgb: np.ndarray) -> str:
    import cv2, base64
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr)
    return base64.b64encode(buf).decode("utf-8")


def visualize_data_point(
    data: Dict[str, Any],
    image_keys: List[str],
    base_prompt: str,
    vlm_output: Optional[List[str]],
    real_response_list: List[Dict[str, Any]],
    filter_json: Dict[str, Any],
    selected_response: List[Dict[str, Any]],
    frame_bbox_map: Dict[int, Dict[str, List[str]]],
    output_dir: str,
    data_idx: int,
    fps: int = 10,
) -> None:
    """Create an MP4 video (one per camera) and a self-contained HTML page for one data point."""
    demo_key = data["demo_key"]
    start_idx = int(data["start_idx"].item() if hasattr(data["start_idx"], "item") else data["start_idx"])
    end_idx   = int(data["end_idx"].item()   if hasattr(data["end_idx"],   "item") else data["end_idx"])
    slug = f"{data_idx:04d}_{demo_key}_{start_idx}_{end_idx}".replace("/", "-").replace(" ", "_")

    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Relative timestep array: maps array position → frame label shown to VLM
    rel_timestep = data["relative_timestep_index"]
    if hasattr(rel_timestep, "cpu"):
        rel_timestep = rel_timestep.cpu().numpy()
    rel_timestep = np.array(rel_timestep)

    # ── build annotated MP4s ──────────────────────────────────────────────────
    # Videos are saved to disk AND embedded as base64 so the HTML is self-contained
    # (required for VSCode Simple Browser / Live Preview which cannot serve local files).
    video_b64_list: List[tuple] = []  # (cam_name, b64_str)
    cameras_frames: Dict[str, np.ndarray] = {}
    for img_key in image_keys:
        frames = data[img_key]
        if hasattr(frames, "cpu"):
            frames = frames.cpu().numpy().astype(np.uint8)
        cameras_frames[img_key] = frames
        T = len(frames)
        annotated = np.stack([
            draw_frame_number(frames[t], int(rel_timestep[t]) if t < len(rel_timestep) else t)
            for t in range(T)
        ])
        cam_name = img_key.replace("/", "_").replace("obs_", "").replace("_image", "")
        mp4_path = os.path.join(video_dir, f"{slug}_{cam_name}.mp4")
        save_video_mp4(annotated, mp4_path, fps=fps)
        video_b64_list.append((cam_name, video_to_b64(mp4_path)))

    # ── extract selected-frame images as base64 ───────────────────────────────
    selected_frame_imgs: Dict[int, Dict[str, str]] = {}
    for res in selected_response:
        rfi = res.get("rel_frame_index")
        if rfi is None or rfi in selected_frame_imgs:
            continue
        positions = np.where(rel_timestep == rfi)[0]
        if len(positions) == 0:
            continue
        pos = int(positions[0])
        selected_frame_imgs[rfi] = {}
        for img_key in image_keys:
            frames = cameras_frames[img_key]
            if pos < len(frames):
                selected_frame_imgs[rfi][img_key] = frame_to_b64_jpeg(frames[pos])

    # ── HTML helpers ──────────────────────────────────────────────────────────
    def _video_tag(cam_name: str, b64: str) -> str:
        return (
            f'<div style="flex:1;min-width:220px;">'
            f'<p style="margin:2px 0;font-size:0.75em;color:#666;">{cam_name}</p>'
            f'<video controls loop style="max-width:100%;border-radius:6px;">'
            f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
            f'Your browser does not support video.</video></div>'
        )

    def _badge(text: str, color: str) -> str:
        return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-size:0.75em;font-weight:bold;">{text}</span>')

    videos_html = "".join(_video_tag(name, b64) for name, b64 in video_b64_list)

    prompt_html = (
        f'<pre style="background:#f5f5f5;padding:12px;border-radius:6px;'
        f'overflow:auto;max-height:320px;font-size:0.78em;white-space:pre-wrap;">'
        f'{html_escape(base_prompt)}</pre>'
    )

    # Raw VLM output
    vlm_html = ""
    for i, out in enumerate(vlm_output or []):
        vlm_html += (
            f'<h4 style="margin:10px 0 4px;color:#555;">VLM Output #{i+1}</h4>'
            f'<pre style="background:#fff3e0;padding:10px;border-radius:6px;'
            f'overflow:auto;max-height:400px;font-size:0.78em;white-space:pre-wrap;">'
            f'{html_escape(out)}</pre>'
        )
    if not vlm_html:
        vlm_html = '<p style="color:#888;">No VLM output recorded.</p>'

    # Filter map: id → filter entry
    filter_map: Dict[str, Dict] = {}
    if filter_json and "results" in filter_json:
        for fr in filter_json["results"]:
            filter_map[str(fr.get("id"))] = fr

    score_color_map = {5: "#2e7d32", 4: "#558b2f", 3: "#f57f17", 2: "#e65100", 1: "#b71c1c"}

    # QA table (all generated items + filter scores)
    qa_rows = ""
    for res in real_response_list:
        rid = res.get("id", "?")
        query     = html_escape(res.get("query", res.get("question", "")))
        answer    = html_escape(res.get("answer", ""))
        task_inst = res.get("task-instruction", "")
        desc      = res.get("description-of-query", "")
        fi        = filter_map.get(str(rid), {})
        score     = fi.get("score", "—")
        decision  = fi.get("decision", "—")
        sc  = score_color_map.get(int(score) if str(score).isdigit() else 0, "#888")
        bc  = "#2e7d32" if decision == "keep" else "#c62828" if decision == "skip" else "#888"
        extra = ""
        if task_inst:
            extra += f'<div style="font-size:0.8em;color:#555;"><b>Instruction:</b> {html_escape(task_inst)}</div>'
        if desc:
            extra += f'<div style="font-size:0.8em;color:#555;"><b>Type:</b> {html_escape(str(desc))}</div>'
        qa_rows += (
            f'<tr>'
            f'<td style="padding:6px;text-align:center;">{rid}</td>'
            f'<td style="padding:6px;">{query}{extra}</td>'
            f'<td style="padding:6px;">{answer}</td>'
            f'<td style="padding:6px;text-align:center;color:{sc};font-weight:bold;">{score}</td>'
            f'<td style="padding:6px;text-align:center;">{_badge(str(decision).upper(), bc)}</td>'
            f'</tr>'
        )
    qa_table_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.88em;margin-top:8px;">'
        f'<thead><tr style="background:#f0f0f0;">'
        f'<th style="padding:6px;min-width:40px;">ID</th>'
        f'<th style="padding:6px;text-align:left;">Query</th>'
        f'<th style="padding:6px;text-align:left;">Answer</th>'
        f'<th style="padding:6px;min-width:60px;">Score</th>'
        f'<th style="padding:6px;min-width:80px;">Decision</th>'
        f'</tr></thead><tbody>{qa_rows}</tbody></table>'
    )

    # Selected frames with QA
    sel_html = ""
    for res in sorted(selected_response, key=lambda x: x.get("rel_frame_index", 0)):
        rfi      = res.get("rel_frame_index", "?")
        query    = html_escape(res.get("query", res.get("question", "")))
        answer   = html_escape(res.get("answer", ""))
        decision = res.get("decision", "")
        score    = res.get("score", "")
        bc       = "#2e7d32" if decision == "keep" else "#c62828"
        imgs_html = ""
        if isinstance(rfi, (int, float)) and int(rfi) in selected_frame_imgs:
            for img_key, b64 in selected_frame_imgs[int(rfi)].items():
                label = img_key.split("/")[-1].replace("_image", "")
                imgs_html += (
                    f'<div><p style="margin:2px 0;font-size:0.72em;color:#666;">{label}</p>'
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="max-width:200px;border-radius:4px;"></div>'
                )
        # bbox table for this frame (exact values as sent to the VLM)
        bbox_html = ""
        if isinstance(rfi, (int, float)) and int(rfi) in frame_bbox_map:
            for cam_name, entries in frame_bbox_map[int(rfi)].items():
                if not entries:
                    continue
                rows = "".join(
                    f'<tr><td style="padding:2px 6px;">{html_escape(e.rsplit("(",1)[0].strip())}</td>'
                    f'<td style="padding:2px 6px;font-family:monospace;color:#444;">'
                    f'{html_escape(e.rsplit("(",1)[1].rstrip(")") if "(" in e else "")}</td></tr>'
                    for e in entries
                )
                bbox_html += (
                    f'<p style="margin:6px 0 2px;font-size:0.78em;color:#555;"><b>BBoxes ({cam_name}):</b></p>'
                    f'<table style="font-size:0.78em;border-collapse:collapse;width:100%;">'
                    f'<thead><tr style="background:#f0f0f0;">'
                    f'<th style="padding:2px 6px;text-align:left;">Object</th>'
                    f'<th style="padding:2px 6px;text-align:left;">x1&nbsp;y1&nbsp;x2&nbsp;y2</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table>'
                )

        sel_html += (
            f'<div style="border:1px solid #ddd;border-radius:6px;padding:10px;margin-bottom:10px;">'
            f'<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;">'
            f'<div style="display:flex;gap:8px;flex-wrap:wrap;">{imgs_html}</div>'
            f'<div style="flex:1;min-width:220px;">'
            f'<p><b>Frame:</b> {rfi} &nbsp; <b>Score:</b> {score} &nbsp;{_badge(str(decision).upper(), bc)}</p>'
            f'<p><b>Q:</b> {query}</p>'
            f'<p><b>A:</b> {answer}</p>'
            f'{bbox_html}'
            f'</div></div></div>'
        )
    if not sel_html:
        sel_html = '<p style="color:#888;">No selected frames.</p>'

    # ── assemble HTML page ────────────────────────────────────────────────────
    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QA Viz: {slug}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #fafafa; color: #222; }}
    h3 {{ margin: 20px 0 6px; border-bottom: 1px solid #ddd; padding-bottom: 4px; color: #333; }}
  </style>
</head>
<body>
  <h2>Data Point: {slug}</h2>
  <p><b>HDF5:</b> {html_escape(data['hdf5_path'])} &nbsp;|&nbsp;
     <b>Demo:</b> {html_escape(demo_key)} &nbsp;|&nbsp;
     <b>Frames:</b> {start_idx}–{end_idx}</p>

  <h3>Videos (with frame numbers)</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;">{videos_html}</div>

  <h3>Text Prompt</h3>
  {prompt_html}

  <h3>VLM Raw Output</h3>
  {vlm_html}

  <h3>Generated QA ({len(real_response_list)} items)</h3>
  {qa_table_html}

  <h3>Selected Frames with QA ({len(selected_response)} items)</h3>
  {sel_html}
</body>
</html>"""

    html_path = os.path.join(output_dir, f"{slug}.html")
    with open(html_path, "w") as fh:
        fh.write(page_html)
    print(f"Visualization: file://{os.path.abspath(html_path)}")
