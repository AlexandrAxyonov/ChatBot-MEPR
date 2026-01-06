from __future__ import annotations

import gradio as gr

from app.analysis import (
    generate_llm_recommendation,
    list_demo_videos,
    load_demo_video,
    reset_analysis_only,
    run_and_show,
)
from app.chat import (
    filter_professions,
    generate_questions_ui,
    ready_to_start,
    regenerate_questions_ui,
    reset_session,
    select_profession,
)
from app.config import DEFAULT_CHECKPOINT
from app.progress import update_history
from core.matching import list_professions


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MEPR Demo", theme=gr.themes.Default()) as demo:
        with gr.Column(elem_id="page_wrap"):
            gr.Markdown("# HR Asisstant")

            demo_video_paths = list_demo_videos()
            profession_choices = list_professions()
            questions_state = gr.State([])
            payload_state = gr.State({})
            history_state = gr.State([])
            session_state = gr.State("")
            viz_state = gr.State({})
            profession_state = gr.State(profession_choices)

            with gr.Column(elem_id="question_panel"):
                gr.Markdown("## Step 1: Select profession and language")
                lang_dd = gr.Dropdown(
                    choices=["English", "Russian"],
                    value="English",
                    label="Language",
                )
                job_title_input = gr.Textbox(
                    label="Profession",
                    placeholder="Start typing to see suggestions...",
                )
                job_suggestions = gr.Radio(
                    choices=[],
                    label="Suggestions",
                    interactive=True,
                )
                questions_status = gr.Markdown("")
                questions_loader = gr.HTML(
                    "<div class='loader'>Generating questions...</div>",
                    visible=False,
                    elem_id="questions_loader",
                )
                questions_md = gr.Markdown("")
                with gr.Row():
                    btn_generate = gr.Button("Generate questions", variant="primary")
                    btn_regen = gr.Button("Regenerate questions", visible=False)
                    btn_ready = gr.Button("Ready to start", visible=False)
                    btn_reset = gr.Button("Start new session", variant="primary", visible=False, elem_id="btn_reset")

            with gr.Column(visible=False, elem_id="analysis_panel") as analysis_panel:
                gr.Markdown("## Step 2: Record your video answers")
                with gr.Row(elem_id="input_panel"):
                    with gr.Column(scale=4, min_width=480, elem_id="video_col"):
                        webcam_constraints = {
                            "width": {"ideal": 640},
                            "height": {"ideal": 360},
                            "frameRate": {"ideal": 24, "max": 30},
                        }
                        in_video = gr.Video(
                            label="Video",
                            elem_id="main_video",
                            include_audio=True,
                            height=420,
                            webcam_options=gr.WebcamOptions(constraints=webcam_constraints),
                        )
                    with gr.Column(scale=2, min_width=220, elem_id="demo_col"):
                        if demo_video_paths:
                            gr.Markdown("### Examples")
                            with gr.Column(elem_id="demo_buttons"):
                                for idx, path in enumerate(demo_video_paths, start=1):
                                    label = f"Example {idx}"
                                    btn = gr.Button(label, variant="secondary", size="sm")
                                    btn.click(fn=load_demo_video, inputs=gr.State(path), outputs=in_video)
                        else:
                            gr.Markdown(
                                "No files found in the `demo_video` folder. "
                                "Create the folder next to `app.py` and put some videos there."
                            )

                in_ckpt = gr.Textbox(
                    label="Checkpoint path",
                    value=DEFAULT_CHECKPOINT,
                    visible=False,
                )
                in_device = gr.Dropdown(
                    label="Device",
                    choices=["auto (select automatically)", "cuda", "cpu"],
                    value="auto (select automatically)",
                    visible=False,
                )
                in_seglen = gr.Slider(
                    5,
                    60,
                    value=30,
                    step=1,
                    label="Segment length (sec.)",
                    visible=False,
                )
                in_outdir = gr.Textbox(
                    label="Output directory",
                    value="outputs",
                    visible=False,
                )
                in_targetfeat = gr.Slider(
                    8,
                    128,
                    value=16,
                    step=2,
                    label="target_features",
                    visible=False,
                )
                in_inputs = gr.Dropdown(
                    label="Attribution source",
                    choices=["features", "emotion_logits", "personality_scores"],
                    value="features",
                    visible=False,
                )

                with gr.Row(elem_id="run_row"):
                    btn_run = gr.Button("Run", variant="primary", elem_id="run_btn")
                status_md = gr.Markdown("", elem_id="status_text")
                runtime_md = gr.Markdown("", elem_id="runtime_text")
                limit_md = gr.Markdown("", elem_id="limit_md", visible=False)

                with gr.Column(visible=False, elem_id="analysis_outputs") as analysis_outputs:
                    out_json = gr.JSON(label="Model outputs (JSON)", elem_id="json_block")

                    with gr.Accordion("Visualization", open=False, elem_id="viz_panel"):
                        @gr.render(inputs=viz_state)
                        def render_visualization(viz):
                            if not isinstance(viz, dict) or not viz:
                                return

                            with gr.Column(elem_id="results_group"):
                                with gr.Row(elem_id="gallery_row"):
                                    with gr.Column(scale=1, min_width=0):
                                        gr.Markdown("### **Body - key gesture frames**")
                                        gr.HTML(value=viz.get("body_html", ""))
                                    with gr.Column(scale=1, min_width=0):
                                        gr.Markdown("### **Face - key expression frames**")
                                        gr.HTML(value=viz.get("face_html", ""))
                                    with gr.Column(scale=1, min_width=0):
                                        gr.Markdown("### **Scene - key context frames**")
                                        gr.HTML(value=viz.get("scene_html", ""))

                                gr.Markdown("", elem_classes=["divider"])
                                with gr.Row(elem_id="audio_bars_row"):
                                    with gr.Column(scale=2, min_width=0, elem_id="audio_text_col"):
                                        gr.Markdown("## Input audio and text data")
                                        gr.Markdown("<div align='center'><h4>Audio waveform</h4></div>")
                                        gr.Image(
                                            value=viz.get("osc_path"),
                                            label="Waveform",
                                            elem_id="osc_img",
                                            interactive=False,
                                            container=False,
                                            height=160,
                                        )
                                        gr.Markdown("<div align='center'><h4>Text transcript</h4></div>")
                                        gr.Textbox(
                                            value=viz.get("transcript", ""),
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
                                                gr.Image(
                                                    value=viz.get("bars_png"),
                                                    label="Emotion Probabilities (Bars)",
                                                    container=False,
                                                    height=200,
                                                    elem_id="emo_bar",
                                                )
                                            with gr.Column(scale=1, min_width=0):
                                                gr.Markdown(
                                                    "<div align='center'><h4>Personality Traits Scores</h4></div>"
                                                )
                                                gr.Image(
                                                    value=viz.get("pers_bars_png"),
                                                    label="Personality Traits Scores (Bars)",
                                                    container=False,
                                                    height=230,
                                                    elem_id="pers_bar",
                                                )

                                gr.Markdown("", elem_classes=["divider"])
                                gr.Markdown("## Visualization of Attention")
                                with gr.Row(elem_classes=["viz_row", "heat_row"]):
                                    gr.Image(
                                        value=viz.get("heatmap"),
                                        label=None,
                                        show_label=False,
                                        container=False,
                                        elem_classes=["heat_img"],
                                        height=240,
                                    )

                                gr.Markdown("", elem_classes=["divider"])
                                gr.Markdown("## Summary")
                                gr.Markdown(value=viz.get("explain_md", ""), label="Explanation")

                    gr.Markdown("## Recommendation", elem_id="recommendation_title")
                    gr.Markdown("Recommendation will be generated automatically after analysis.")
                    llm_loader = gr.HTML(
                        "<div class='loader'>Generating recommendation...</div>",
                        visible=False,
                        elem_id="llm_loader",
                    )
                    out_llm = gr.Chatbot(value=[], height=320, label="AI Recommendation")
                    btn_explain = gr.Button(
                        "Regenerate recommendation",
                        variant="secondary",
                        visible=False,
                    )
                    btn_try_again = gr.Button(
                        "Try another video",
                        variant="secondary",
                        visible=False,
                        elem_id="btn_try_again",
                    )

                with gr.Column(visible=False, elem_id="progress_panel") as progress_panel:
                    gr.Markdown("## Progress")
                    progress_md = gr.Markdown("", elem_id="progress_md")
                    with gr.Row(elem_id="radar_row"):
                        radar_plot = gr.Plot(label="Trait profile progress", elem_id="radar_plot", visible=False)
                    progress_bar = gr.Plot(label="Trait comparison", elem_id="progress_bar", visible=False)
                    gr.Markdown(
                        "O=Openness, C=Conscientiousness, E=Extraversion, A=Agreeableness, N=Non-Neuroticism.",
                        elem_id="ocean_note",
                    )

        btn_generate.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[questions_loader],
        ).then(
            fn=generate_questions_ui,
            inputs=[job_title_input, lang_dd],
            outputs=[
                questions_state,
                questions_md,
                questions_status,
                btn_ready,
                btn_regen,
                questions_loader,
            ],
            show_progress="hidden",
        )

        job_title_input.input(
            fn=filter_professions,
            inputs=[job_title_input, profession_state, job_suggestions],
            outputs=[job_suggestions],
        )
        job_title_input.change(
            fn=filter_professions,
            inputs=[job_title_input, profession_state, job_suggestions],
            outputs=[job_suggestions],
        )
        job_title_input.submit(
            fn=filter_professions,
            inputs=[job_title_input, profession_state, job_suggestions],
            outputs=[job_suggestions],
        )

        job_suggestions.change(
            fn=select_profession,
            inputs=[job_suggestions],
            outputs=[job_title_input],
        )

        btn_regen.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[questions_loader],
        ).then(
            fn=regenerate_questions_ui,
            inputs=[job_title_input, lang_dd],
            outputs=[
                questions_state,
                questions_md,
                questions_status,
                btn_ready,
                btn_regen,
                questions_loader,
            ],
            show_progress="hidden",
        )

        btn_ready.click(
            fn=ready_to_start,
            inputs=[],
            outputs=[
                analysis_panel,
                btn_generate,
                btn_ready,
                btn_regen,
                btn_reset,
                analysis_outputs,
                job_title_input,
                job_suggestions,
                lang_dd,
            ],
        )

        btn_reset.click(
            fn=reset_session,
            inputs=[],
            outputs=[
                lang_dd,
                job_title_input,
                job_suggestions,
                questions_state,
                questions_md,
                questions_status,
                questions_loader,
                btn_generate,
                btn_ready,
                btn_regen,
                btn_reset,
                analysis_panel,
                analysis_outputs,
                in_video,
                out_json,
                out_llm,
                llm_loader,
                btn_explain,
                payload_state,
                session_state,
                history_state,
                viz_state,
                status_md,
                runtime_md,
                limit_md,
                btn_run,
                progress_md,
                progress_panel,
                btn_try_again,
                radar_plot,
                progress_bar,
            ],
        )

        btn_run.click(
            fn=lambda: (
                gr.update(value="Processing...", visible=True),
                "",
                gr.update(visible=False),
                [],
                gr.update(visible=False),
                gr.update(visible=False),
                {},
            ),
            inputs=[],
            outputs=[
                status_md,
                runtime_md,
                analysis_outputs,
                out_llm,
                btn_explain,
                llm_loader,
                viz_state,
            ],
        ).then(
            fn=run_and_show,
            inputs=[
                job_title_input,
                in_video,
                in_ckpt,
                in_device,
                in_seglen,
                in_outdir,
                in_targetfeat,
                in_inputs,
                session_state,
                history_state,
            ],
            outputs=[
                out_json,
                payload_state,
                viz_state,
                status_md,
                runtime_md,
                analysis_outputs,
                session_state,
            ],
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[llm_loader],
        ).then(
            fn=generate_llm_recommendation,
            inputs=[payload_state, job_title_input, lang_dd],
            outputs=[out_json, payload_state, out_llm],
            show_progress="hidden",
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            inputs=[],
            outputs=[llm_loader, btn_explain],
        ).then(
            fn=update_history,
            inputs=[payload_state, job_title_input, lang_dd, history_state],
            outputs=[
                history_state,
                progress_md,
                progress_panel,
                btn_try_again,
                btn_run,
                limit_md,
                radar_plot,
                progress_bar,
            ],
            show_progress="hidden",
        )

        btn_try_again.click(
            fn=reset_analysis_only,
            inputs=[],
            outputs=[
                in_video,
                out_json,
                out_llm,
                llm_loader,
                btn_explain,
                payload_state,
                status_md,
                runtime_md,
                analysis_outputs,
                viz_state,
            ],
        )

        btn_explain.click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[llm_loader],
        ).then(
            fn=generate_llm_recommendation,
            inputs=[payload_state, job_title_input, lang_dd],
            outputs=[out_json, payload_state, out_llm],
            show_progress="hidden",
        ).then(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=[llm_loader],
        )

    return demo
