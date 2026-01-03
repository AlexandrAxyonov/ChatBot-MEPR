from __future__ import annotations

import os

import gradio as gr

from app.analysis import (
    generate_llm_recommendation,
    list_demo_videos,
    load_demo_video,
    run_and_show,
)
from app.chat import chat_submit, ready_to_start, regenerate_questions, reset_session
from app.config import DEFAULT_CHECKPOINT


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MEPR Demo") as demo:
        with gr.Column(elem_id="page_wrap"):
            gr.Markdown("# HR Asisstant")

            demo_video_paths = list_demo_videos()
            stage_state = gr.State("ask_title")
            job_title_state = gr.State("")
            job_desc_state = gr.State("")
            questions_state = gr.State([])
            payload_state = gr.State({})

            with gr.Group(elem_id="question_panel"):
                gr.Markdown("## Chat")
                chatbot = gr.Chatbot(
                    value=[
                        {
                            "role": "assistant",
                            "content": (
                                "Hi! Please tell me which position you are applying for."
                            ),
                        }
                    ],
                    height=320,
                    layout="panel",
                )
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message and press Enter",
                )
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    btn_regen = gr.Button("Regenerate questions", visible=False)
                    btn_ready = gr.Button("Ready to start", visible=False)

            with gr.Group(visible=False, elem_id="analysis_panel") as analysis_panel:
                gr.Markdown("## Step 2: Record your video answers")
                with gr.Row():
                    btn_reset = gr.Button("Start new session", variant="primary", elem_id="btn_reset")
                with gr.Row(elem_id="input_panel"):
                    with gr.Column(scale=4, min_width=500, elem_id="video_col"):
                        webcam_constraints = {
                            "width": {"ideal": 640},
                            "height": {"ideal": 360},
                            "frameRate": {"ideal": 24, "max": 30},
                        }
                        in_video = gr.Video(
                            label="Video",
                            elem_id="main_video",
                            include_audio=True,
                            webcam_options=gr.WebcamOptions(constraints=webcam_constraints),
                        )

                    with gr.Column(scale=3, min_width=420, elem_id="settings_col"):
                        with gr.Accordion("Advanced settings", open=False):
                            in_ckpt = gr.Textbox(label="Checkpoint path", value=DEFAULT_CHECKPOINT)
                            in_device = gr.Dropdown(
                                label="Device",
                                choices=["auto (select automatically)", "cuda", "cpu"],
                                value="auto (select automatically)",
                            )
                            in_seglen = gr.Slider(5, 60, value=30, step=1, label="Segment length (sec.)")
                            in_outdir = gr.Textbox(label="Output directory", value="outputs")

                        with gr.Accordion("Visualization settings", open=False):
                            in_targetfeat = gr.Slider(8, 128, value=16, step=2, label="target_features")
                            in_inputs = gr.Dropdown(
                                label="Attribution source",
                                choices=["features", "emotion_logits", "personality_scores"],
                                value="features",
                            )

                with gr.Row(elem_id="demo_heat_row"):
                    with gr.Column(scale=4, min_width=500, elem_id="demo_col"):
                        if demo_video_paths:
                            gr.Markdown("### Or select one of the preloaded videos:")
                            with gr.Row(elem_id="demo_row"):
                                for path in demo_video_paths:
                                    label = os.path.basename(path)
                                    btn = gr.Button(label, variant="secondary", size="sm")
                                    btn.click(fn=load_demo_video, inputs=gr.State(path), outputs=in_video)
                        else:
                            gr.Markdown(
                                "No files found in the `demo_video` folder. "
                                "Create the folder next to `app.py` and put some videos there."
                            )

                with gr.Row():
                    btn_run = gr.Button("Run", variant="primary", elem_id="run_btn")
                status_md = gr.Markdown("", elem_id="status_text")
                runtime_md = gr.Markdown("", elem_id="runtime_text")

                with gr.Group(visible=False, elem_id="analysis_outputs") as analysis_outputs:
                    out_json = gr.JSON(label="Model outputs (JSON)", elem_id="json_block")

                    gr.Markdown("## Visualization", elem_id="viz_panel")
                    gr.Markdown("", elem_classes=["divider"])
                    with gr.Column(elem_id="results_wrapper"):
                        with gr.Group(visible=False, elem_id="results_group") as results_group:
                            with gr.Row(elem_id="gallery_row"):
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("### **Body - key gesture frames**")
                                    out_body = gr.HTML()
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("### **Face - key expression frames**")
                                    out_face = gr.HTML()
                                with gr.Column(scale=1, min_width=0):
                                    gr.Markdown("### **Scene - key context frames**")
                                    out_scene = gr.HTML()

                            gr.Markdown("", elem_classes=["divider"])
                            with gr.Row(elem_id="audio_bars_row"):
                                with gr.Column(scale=2, min_width=0, elem_id="audio_text_col"):
                                    gr.Markdown("## Input audio and text data")
                                    gr.Markdown("<div align='center'><h4>Audio waveform</h4></div>")
                                    out_osc = gr.Image(
                                        label="Waveform",
                                        elem_id="osc_img",
                                        interactive=False,
                                        container=False,
                                        height=160,
                                    )
                                    gr.Markdown("<div align='center'><h4>Text transcript</h4></div>")
                                    out_transcript = gr.Textbox(
                                        label=None,
                                        show_label=False,
                                        lines=3,
                                        container=False,
                                        elem_id="transcript_box",
                                    )

                                with gr.Column(scale=5, min_width=0, elem_id="bars_col"):
                                    gr.Markdown("## Prediction result")
                                    with gr.Row(elem_id="bars_row_inner"):
                                        with gr.Column(scale=1, min_width=0):
                                            gr.Markdown(
                                                "<div align='center'><h4>Probability distribution of emotions</h4></div>"
                                            )
                                            out_bars_png = gr.Image(
                                                label="Emotion Probabilities (Bars)",
                                                container=False,
                                                height=200,
                                                elem_id="emo_bar",
                                            )
                                        with gr.Column(scale=1, min_width=0):
                                            gr.Markdown(
                                                "<div align='center'><h4>Personality Traits Scores</h4></div>"
                                            )
                                            out_pers_bars_png = gr.Image(
                                                label="Personality Traits Scores (Bars)",
                                                container=False,
                                                height=230,
                                                elem_id="pers_bar",
                                            )

                            gr.Markdown("", elem_classes=["divider"])
                            gr.Markdown("## Visualization of Attention")
                            with gr.Row(elem_classes=["viz_row", "heat_row"]):
                                out_heat_all = gr.Image(
                                    label=None,
                                    show_label=False,
                                    container=False,
                                    elem_classes=["heat_img"],
                                    height=240,
                                )

                            gr.Markdown("", elem_classes=["divider"])
                            gr.Markdown("## Summary")
                            out_explain = gr.Markdown(label="Explanation")

                    gr.Markdown("## Recommendation", elem_id="recommendation_title")
                    gr.Markdown("Want a tailored explanation? Click the button below.")
                    btn_explain = gr.Button("Generate recommendation", variant="primary")
                    out_llm = gr.Chatbot(value=[], height=320, label="AI Recommendation")

        send_btn.click(
            fn=chat_submit,
            inputs=[user_input, chatbot, stage_state, job_title_state, job_desc_state, questions_state],
            outputs=[
                chatbot,
                stage_state,
                job_title_state,
                job_desc_state,
                questions_state,
                btn_ready,
                btn_regen,
                user_input,
            ],
        )
        user_input.submit(
            fn=chat_submit,
            inputs=[user_input, chatbot, stage_state, job_title_state, job_desc_state, questions_state],
            outputs=[
                chatbot,
                stage_state,
                job_title_state,
                job_desc_state,
                questions_state,
                btn_ready,
                btn_regen,
                user_input,
            ],
        )

        btn_regen.click(
            fn=regenerate_questions,
            inputs=[chatbot, job_title_state, job_desc_state],
            outputs=[chatbot, questions_state, btn_ready, btn_regen],
        )

        btn_ready.click(
            fn=ready_to_start,
            inputs=[],
            outputs=[
                stage_state,
                analysis_panel,
                btn_ready,
                btn_regen,
                user_input,
                send_btn,
                analysis_outputs,
            ],
        )

        btn_reset.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                chatbot,
                stage_state,
                job_title_state,
                job_desc_state,
                questions_state,
                btn_ready,
                btn_regen,
                analysis_panel,
                user_input,
                send_btn,
                in_video,
                out_json,
                out_llm,
                payload_state,
                status_md,
                results_group,
                runtime_md,
                analysis_outputs,
                out_transcript,
                out_osc,
                out_bars_png,
                out_heat_all,
                out_explain,
                out_body,
                out_face,
                out_scene,
                out_pers_bars_png,
            ],
        )

        btn_run.click(
            fn=lambda: (
                gr.update(value="Processing...", visible=True),
                gr.update(visible=False),
                "",
                gr.update(visible=False),
            ),
            inputs=[],
            outputs=[status_md, results_group, runtime_md, analysis_outputs],
        ).then(
            fn=run_and_show,
            inputs=[
                job_title_state,
                job_desc_state,
                in_video,
                in_ckpt,
                in_device,
                in_seglen,
                in_outdir,
                in_targetfeat,
                in_inputs,
            ],
            outputs=[
                out_json,
                payload_state,
                out_transcript,
                out_osc,
                out_bars_png,
                out_heat_all,
                out_explain,
                out_body,
                out_face,
                out_scene,
                out_pers_bars_png,
                status_md,
                results_group,
                runtime_md,
                analysis_outputs,
            ],
        )

        btn_explain.click(
            fn=generate_llm_recommendation,
            inputs=[payload_state, job_title_state, job_desc_state],
            outputs=[out_json, payload_state, out_llm],
        )

    return demo
