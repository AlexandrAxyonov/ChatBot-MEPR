from __future__ import annotations

from typing import List

import gradio as gr

from app.config import DEFAULT_NUM_QUESTIONS
from core.llm.qwen_client import generate_questions, unload_model


def format_questions_markdown(questions: List[str]) -> str:
    if not questions:
        return ""
    lines = [f"{i + 1}. {q}" for i, q in enumerate(questions)]
    return "\n".join(lines)


def _append_chat(history: List[dict] | None, role: str, content: str) -> List[dict]:
    history = history or []
    history.append({"role": role, "content": content})
    return history


def chat_submit(
    user_message: str,
    history: List[dict] | None,
    stage: str,
    job_title: str,
    job_description: str,
    questions: List[str],
):
    message = (user_message or "").strip()
    if not message:
        return (
            history or [],
            stage,
            job_title,
            job_description,
            questions,
            gr.update(visible=stage == "questions_ready" and bool(questions)),
            gr.update(visible=stage == "questions_ready"),
            "",
        )

    if stage == "ask_title":
        job_title = message
        history = _append_chat(history, "user", message)
        history = _append_chat(
            history,
            "assistant",
            "Thanks. Now send a job description or a short resume summary.",
        )
        return (
            history,
            "ask_desc",
            job_title,
            job_description,
            questions,
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )

    if stage == "ask_desc":
        job_description = message
        history = _append_chat(history, "user", message)
        questions = generate_questions(
            job_title=job_title,
            job_description=job_description,
            num_questions=DEFAULT_NUM_QUESTIONS,
        )
        if questions:
            questions_text = format_questions_markdown(questions)
            history = _append_chat(
                history,
                "assistant",
                "Here are the questions for your video answers:\n"
                f"{questions_text}\n"
                "Click Ready to start when you are set.",
            )
            return (
                history,
                "questions_ready",
                job_title,
                job_description,
                questions,
                gr.update(visible=True),
                gr.update(visible=True),
                "",
            )
        history = _append_chat(
            history,
            "assistant",
            "Could not generate questions. You can edit the description and send again, "
            "or click Regenerate questions.",
        )
        return (
            history,
            "ask_desc",
            job_title,
            job_description,
            [],
            gr.update(visible=False),
            gr.update(visible=True),
            "",
        )

    history = _append_chat(history, "user", message)
    history = _append_chat(
        history,
        "assistant",
        "Questions are already prepared. Click Ready to start or Regenerate questions.",
    )
    return (
        history,
        stage,
        job_title,
        job_description,
        questions,
        gr.update(visible=bool(questions)),
        gr.update(visible=True),
        "",
    )


def regenerate_questions(
    history: List[dict] | None,
    job_title: str,
    job_description: str,
):
    if not (job_title or job_description):
        history = _append_chat(
            history,
            "assistant",
            "Please provide the job title and description/resume in the chat first.",
        )
        return history, [], gr.update(visible=False), gr.update(visible=False)

    questions = generate_questions(
        job_title=job_title,
        job_description=job_description,
        num_questions=DEFAULT_NUM_QUESTIONS,
    )
    if not questions:
        history = _append_chat(
            history,
            "assistant",
            "Could not generate questions. Please try again.",
        )
        return history, [], gr.update(visible=False), gr.update(visible=True)

    questions_text = format_questions_markdown(questions)
    history = _append_chat(
        history,
        "assistant",
        "New list of questions:\n"
        f"{questions_text}",
    )
    return (
        history,
        questions,
        gr.update(visible=True),
        gr.update(visible=True),
    )


def ready_to_start():
    unload_model()
    return (
        "analysis",
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(interactive=False, value=""),
        gr.update(interactive=False),
        gr.update(visible=False),
    )


def reset_session():
    unload_model()
    history = [
        {
            "role": "assistant",
            "content": "Hi! Please tell me which position you are applying for.",
        }
    ]
    return (
        history,
        "ask_title",
        "",
        "",
        [],
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(interactive=True, value=""),
        gr.update(interactive=True),
        gr.update(value=None),
        {},
        [],
        {},
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        None,
        None,
        None,
        "",
        "",
        "",
        "",
        None,
    )
