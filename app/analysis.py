from __future__ import annotations

import os
import time
from typing import List

import gradio as gr

from app.config import (
    DEFAULT_CHECKPOINT,
    DEMO_DIR,
    EMO_ORDER,
    EMO_ORDER_FOR_BARS,
    PERS_COLORS,
    PERS_ORDER,
    VIDEO_EXTS,
)
from app.utils import build_output_payload, choose_device, convert_webm_to_mp4, ensure_dir, render_gallery_html
from core.attribution import (
    compute_contributions_from_result,
    format_contribution_summary,
    plot_emotion_probs_barchart,
    visualize_all_task_heatmaps,
)
from core.media_utils import extract_keyframes_from_result
from core.matching import match_profession
from core.llm.qwen_client import generate_explanation, unload_model
from core.runtime import analyze_video_basic, get_multitask_pred_with_attribution


def run_basic_and_split_heatmap(
    video_path,
    checkpoint_path,
    device_choice,
    segment_length,
    out_dir,
    target_features,
    inputs_choice,
):
    empty_txt = ""
    if isinstance(video_path, dict):
        video_path = video_path.get("name") or video_path.get("path")
    if video_path is None:
        return (
            {"error": "Please upload a video."},
            empty_txt,
            None,
            None,
            None,
            "Please upload a video.",
            "",
            "",
            "",
            None,
            0.0,
        )

    ckpt = (checkpoint_path or "").strip() or DEFAULT_CHECKPOINT
    dev = choose_device(device_choice)
    save_dir = ensure_dir((out_dir or "").strip() or "outputs")

    if os.path.splitext(video_path)[1].lower() == ".webm":
        video_path = convert_webm_to_mp4(video_path)

    res_basic = analyze_video_basic(
        video_path=video_path,
        checkpoint_path=ckpt,
        segment_length=int(segment_length),
        device=dev,
        save_dir=save_dir,
    )
    video_duration = float(res_basic.get("video_duration_sec", 0.0))

    payload = build_output_payload(res_basic)
    transcript = res_basic.get("transcript", "")
    emo_prob = list(map(float, res_basic.get("emotion_logits", [0.0] * len(EMO_ORDER))))
    per_prob = list(map(float, res_basic.get("personality_scores", [0.0] * len(PERS_ORDER))))

    osc = res_basic.get("oscilloscope_path")
    osc_path = osc if (osc and os.path.exists(osc)) else None

    bars_png = os.path.join(save_dir, "emo_bars.png")
    remap = [EMO_ORDER.index(lbl) for lbl in EMO_ORDER_FOR_BARS]
    emo_prob_for_bars = [emo_prob[i] for i in remap]

    emo_colors = {
        "Neutral": "#9CA3AF",
        "Anger": "#EF4444",
        "Disgust": "#10B981",
        "Fear": "#8B5CF6",
        "Happiness": "#F59E0B",
        "Sadness": "#3B82F6",
        "Surprise": "#EC4899",
    }
    palette = [emo_colors[lbl] for lbl in EMO_ORDER_FOR_BARS]

    plot_emotion_probs_barchart(
        probs=emo_prob_for_bars,
        labels=EMO_ORDER_FOR_BARS,
        out_path=bars_png,
        colors=palette,
        figsize=(8, 3.7),
        auto_ylim=False,
    )

    pers_bars_png = os.path.join(save_dir, "pers_bars.png")
    plot_emotion_probs_barchart(
        probs=per_prob,
        labels=PERS_ORDER,
        out_path=pers_bars_png,
        title="Score (%)",
        colors=[PERS_COLORS[lbl] for lbl in PERS_ORDER],
        figsize=(8, 4),
        auto_ylim=True,
        ylim_margin=3.0,
    )

    res_attr = get_multitask_pred_with_attribution(
        video_path=video_path,
        checkpoint_path=ckpt,
        segment_length=int(segment_length),
        device=dev,
    )

    combined_heatmap = visualize_all_task_heatmaps(
        result=res_attr,
        name_video=os.path.basename(video_path),
        target_features=int(target_features),
        inputs=inputs_choice,
        out_dir="heatmaps_img",
    )

    contrib = compute_contributions_from_result(
        result=res_attr,
        inputs=inputs_choice,
        modality_order=["body", "face", "scene", "audio", "text"],
        target_features=int(target_features),
        visual_granularity="detailed",
    )
    explain_md = format_contribution_summary(contrib, emo_labels=EMO_ORDER)

    sample_dir = os.path.join(save_dir, "samples")
    frames_info = extract_keyframes_from_result(
        video_path=video_path,
        result=res_attr,
        out_dir=sample_dir,
        n_default=8,
    )

    indices = frames_info.get("indices", [])
    body_paths = frames_info.get("body", [])
    face_paths = frames_info.get("face", [])
    scene_paths = frames_info.get("scene", [])

    captions = [f"Frame {idx}" for idx in indices]

    thumb_h = 120
    body_html = render_gallery_html(body_paths, captions, uid="body", thumb_h=thumb_h)
    face_html = render_gallery_html(face_paths, captions, uid="face", thumb_h=thumb_h)
    scene_html = render_gallery_html(scene_paths, captions, uid="scene", thumb_h=thumb_h)

    return (
        payload,
        transcript,
        osc_path,
        bars_png,
        combined_heatmap,
        explain_md,
        body_html,
        face_html,
        scene_html,
        pers_bars_png,
        video_duration,
    )


def list_demo_videos() -> List[str]:
    if not os.path.isdir(DEMO_DIR):
        return []

    files = []
    for name in os.listdir(DEMO_DIR):
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS:
            files.append(os.path.join(DEMO_DIR, name))
    return sorted(files)


def load_demo_video(path: str):
    return path


def run_and_show(
    job_title,
    job_description,
    video_path,
    checkpoint_path,
    device_choice,
    segment_length,
    out_dir,
    target_features,
    inputs_choice,
):
    start = time.time()
    result = run_basic_and_split_heatmap(
        video_path,
        checkpoint_path,
        device_choice,
        segment_length,
        out_dir,
        target_features,
        inputs_choice,
    )
    *core_outputs, video_duration = result
    payload = core_outputs[0]
    if isinstance(payload, dict) and payload.get("error"):
        transcript = core_outputs[1]
        osc_path = core_outputs[2]
        bars_png = core_outputs[3]
        heatmap = core_outputs[4]
        explain_md = core_outputs[5]
        body_html = core_outputs[6]
        face_html = core_outputs[7]
        scene_html = core_outputs[8]
        pers_bars_png = core_outputs[9]
        return (
            payload,
            payload,
            transcript,
            osc_path,
            bars_png,
            heatmap,
            explain_md,
            body_html,
            face_html,
            scene_html,
            pers_bars_png,
            gr.update(value=payload["error"], visible=True),
            gr.update(visible=False),
            "",
            gr.update(visible=False),
        )

    if isinstance(payload, dict):
        user_scores = payload.get("personality", {})
        match_result = match_profession(
            user_scores=user_scores,
            job_title=job_title or "",
            job_description=job_description or "",
        )
        payload["profession_match"] = match_result
        core_outputs[0] = payload

    elapsed = time.time() - start
    runtime_txt = f"Video duration: {video_duration:.1f} sec | Inference time: {elapsed:.1f} sec"
    transcript = core_outputs[1]
    osc_path = core_outputs[2]
    bars_png = core_outputs[3]
    heatmap = core_outputs[4]
    explain_md = core_outputs[5]
    body_html = core_outputs[6]
    face_html = core_outputs[7]
    scene_html = core_outputs[8]
    pers_bars_png = core_outputs[9]
    return (
        payload,
        payload,
        transcript,
        osc_path,
        bars_png,
        heatmap,
        explain_md,
        body_html,
        face_html,
        scene_html,
        pers_bars_png,
        gr.update(value="", visible=False),
        gr.update(visible=True),
        runtime_txt,
        gr.update(visible=True),
    )


def generate_llm_recommendation(payload: dict, job_title: str, job_description: str):
    if not isinstance(payload, dict):
        return payload, payload, []

    llm_text = build_llm_explanation(
        payload=payload,
        job_title=job_title or "",
        job_description=job_description or "",
    )
    if llm_text:
        payload = dict(payload)
        payload["llm_explanation"] = llm_text
    unload_model()
    chat = [{"role": "assistant", "content": llm_text}] if llm_text else []
    return payload, payload, chat


def build_llm_explanation(payload: dict, job_title: str, job_description: str) -> str:
    emotion = payload.get("emotion", {})
    personality = payload.get("personality", {})
    match = payload.get("profession_match", {})

    top_emotion = str(emotion.get("top", ""))
    top_prob = float(emotion.get("top_prob", 0.0))

    traits_sorted = sorted(personality.items(), key=lambda x: x[1], reverse=True)
    highs = traits_sorted[:2]
    lows = traits_sorted[-2:] if len(traits_sorted) >= 2 else traits_sorted

    match_prof = match.get("matched_profession") or {}
    match_source = match.get("match_source") or "unknown"
    similarity = float(match_prof.get("similarity", 0.0))
    top_roles = match.get("top_professions", [])
    top1_sim = float(top_roles[0].get("similarity", similarity)) if top_roles else similarity
    gap = max(0.0, top1_sim - similarity)

    verdict = "a weak fit"
    if top_roles:
        if similarity == top1_sim:
            verdict = "likely a good fit"
        elif gap <= 0.01 and similarity >= 0.85:
            verdict = "likely a good fit"
        elif gap <= 0.03:
            verdict = "a partial fit"
        else:
            verdict = "a weak fit"
    else:
        if similarity >= 0.7:
            verdict = "likely a good fit"
        elif similarity >= 0.6:
            verdict = "a partial fit"

    negatives = {"Anger", "Sadness", "Fear", "Disgust"}
    emotion_warning = bool(top_emotion in negatives and top_prob >= 0.5)

    top_roles = top_roles[:3]
    top_roles_text = ", ".join(
        f"{r.get('profession', '')} ({r.get('similarity', 0.0):.4f})" for r in top_roles
    )

    high_text = ", ".join(f"{k} {v:.2f}" for k, v in highs)
    low_text = ", ".join(f"{k} {v:.2f}" for k, v in lows)

    matched_name = str(match_prof.get("profession", ""))
    if job_title and match_source != "title":
        role_note = (
            f"Requested role '{job_title}' was not found in the dataset. "
            f"Closest match used: '{matched_name}'."
        )
    else:
        role_note = f"Requested role: {job_title or 'N/A'}."

    summary = (
        f"{role_note}\n"
        f"Job description: {job_description}\n"
        f"Verdict: {verdict} (similarity {similarity:.4f}, gap {gap:.2f}).\n"
        f"Top emotion: {top_emotion} ({top_prob:.2f}).\n"
        f"High traits: {high_text}.\n"
        f"Low traits: {low_text}.\n"
        f"Top alternative roles: {top_roles_text}.\n"
        f"Emotion caution: {'yes' if emotion_warning else 'no'}.\n"
    )

    try:
        return generate_explanation(summary=summary)
    except Exception:
        return ""
