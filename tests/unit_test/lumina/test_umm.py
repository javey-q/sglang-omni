# SPDX-License-Identifier: Apache-2.0
"""Lumina's controller decisions, request validation, and image sizing."""

import asyncio
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang_omni.models.lumina import stages
from sglang_omni.models.lumina.config import LuminaPipelineConfig
from sglang_omni.models.lumina.generation import (
    EndEvent,
    IncompleteInterleavedGenerationError,
    IncompleteTextGenerationError,
    TextEvent,
)
from sglang_omni.models.lumina.preprocessor import (
    canonical_image,
    preprocess,
    smart_resize_shape,
)
from sglang_omni.models.lumina.settings import LuminaGenerationParams
from sglang_omni.models.lumina.umm import EVENT, PREPARE, LuminaUMMAdapter
from sglang_omni.pipeline.umm import UMMContext, UMMRequestError
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage


def context_for(
    inputs: object,
    *,
    options: dict[str, object] | None = None,
    remaining_dispatches: int = 10,
) -> UMMContext:
    params = {} if options is None else {"stage_params": {"orchestrator": options}}
    return UMMContext(
        execution_id="execution",
        request=OmniRequest(inputs, params, {}),
        deadline=1e9,
        remaining_dispatches=remaining_dispatches,
    )


USER_TURN = [{"role": "user", "content": "Draw a cat."}]


def test_start_prepares_with_validated_params() -> None:
    adapter = LuminaUMMAdapter("model")
    step = adapter.start(context_for(USER_TURN, options={"height": 192, "seed": 7}))
    assert step.phase == PREPARE
    assert step.request.metadata["lumina_generation"]["height"] == 192
    assert step.request.metadata["lumina_generation"]["seed"] == 7


@pytest.mark.parametrize(
    ("inputs", "options"),
    [
        ([], None),
        ([{"role": "assistant", "content": "hi"}], None),
        ([{"role": "tool", "content": "hi"}], None),
        (
            [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late"},
                {"role": "user", "content": "hi"},
            ],
            None,
        ),
        ({"messages": USER_TURN, "images": ["a.png"]}, None),
        (USER_TURN, {"height": 100}),
        (USER_TURN, {"unknown": 1}),
    ],
)
def test_start_rejects_requests_it_cannot_serve(
    inputs: object, options: dict[str, object] | None
) -> None:
    with pytest.raises(UMMRequestError):
        LuminaUMMAdapter("model").start(context_for(inputs, options=options))


def test_start_rejects_a_budget_the_deployment_cannot_run() -> None:
    context = context_for(USER_TURN, options={"max_events": 20})
    with pytest.raises(UMMRequestError, match="max_dispatches"):
        LuminaUMMAdapter("model").start(context)


def test_prepare_result_starts_the_first_event_with_its_inputs() -> None:
    context = context_for(USER_TURN)
    prepared = {"messages": USER_TURN, "generation": {}}
    step = LuminaUMMAdapter("model").advance(context, PREPARE, prepared)
    assert [dispatch.phase for dispatch in step.dispatches] == [EVENT]
    assert step.dispatches[0].request.inputs == prepared
    assert step.outputs == ()


def test_later_events_carry_no_inputs_and_segments_keep_their_kind() -> None:
    adapter = LuminaUMMAdapter("model")
    context = context_for(USER_TURN)
    step = adapter.advance(
        context,
        EVENT,
        {"output": {"kind": "text", "text": "A cat."}, "finish_reason": None},
    )
    assert [(output.kind, output.data) for output in step.outputs] == [
        ("text", "A cat.")
    ]
    assert step.dispatches[0].request.inputs == {}

    image = {"kind": "image", "url": "data:image/png;base64,AAAA"}
    step = adapter.advance(context, EVENT, {"output": image, "finish_reason": None})
    assert [(output.kind, output.data) for output in step.outputs] == [("image", image)]


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_a_finished_event_completes_with_its_reason_and_empty_text_emits_nothing(
    finish_reason: str,
) -> None:
    adapter = LuminaUMMAdapter("model")
    context = context_for(USER_TURN)
    finished = {"output": None, "finish_reason": finish_reason}
    step = adapter.advance(context, EVENT, finished)
    assert step.dispatches == ()
    assert step.finish_reason == finish_reason
    empty = {"output": {"kind": "text", "text": ""}, "finish_reason": finish_reason}
    assert adapter.advance(context, EVENT, empty).outputs == ()


def test_default_budget_fits_the_default_limits() -> None:
    config = LuminaPipelineConfig(model_path="model")
    controller = config.stage_named("orchestrator")
    default_events = LuminaGenerationParams().max_events
    assert controller.factory.max_dispatches == default_events + 2
    assert controller.factory.max_outputs == default_events + 1


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (512, 512, (256, 256)),
        (300, 400, (192, 256)),
        (400, 300, (256, 192)),
        (10, 4000, (32, 256)),
    ],
)
def test_smart_resize_fixes_the_long_edge(
    height: int, width: int, expected: tuple[int, int]
) -> None:
    assert smart_resize_shape(height, width, long_edge=256) == expected


def test_generated_history_keeps_its_exact_size_and_user_images_resize() -> None:
    aligned = Image.new("RGB", (192, 512))
    assert (
        canonical_image(aligned, role="assistant", long_edge=256)["resized_height"]
        == 512
    )
    item = canonical_image(aligned, role="user", long_edge=256)
    assert (item["resized_height"], item["resized_width"]) == (256, 96)
    assert item["image"].size == (96, 256)

    unaligned = Image.new("RGB", (100, 100))
    item = canonical_image(unaligned, role="assistant", long_edge=256)
    assert item["image"].size == (256, 256)


def test_preprocess_canonicalizes_messages_and_adds_the_system_turn() -> None:
    image = Image.new("RGB", (64, 64))
    request = OmniRequest(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Edit it."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "image", "image": image}, "Done."],
            },
            {"role": "user", "content": "Again."},
        ],
        {},
        {"lumina_generation": LuminaGenerationParams(seed=3).model_dump()},
    )
    payload = asyncio.run(preprocess(StagePayload("request", request, {})))
    messages = payload.data["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"][0]["image"].size == (256, 256)
    assert messages[2]["content"][0]["image"].size == (64, 64)
    assert messages[2]["content"][1] == {"type": "text", "text": "Done."}
    assert messages[3]["content"] == [{"type": "text", "text": "Again."}]
    assert payload.data["generation"]["seed"] == 3


def test_preprocess_keeps_one_system_turn_when_the_client_sent_its_own() -> None:
    request = OmniRequest(
        [{"role": "system", "content": "Be brief."}, *USER_TURN],
        {},
        {"lumina_generation": LuminaGenerationParams().model_dump()},
    )
    payload = asyncio.run(preprocess(StagePayload("request", request, {})))
    roles = [message["role"] for message in payload.data["messages"]]
    assert roles == ["system", "user"]


def test_preprocess_rejects_parts_the_model_cannot_read() -> None:
    request = OmniRequest(
        [{"role": "user", "content": [{"type": "video", "video": "a.mp4"}]}],
        {},
        {"lumina_generation": LuminaGenerationParams().model_dump()},
    )
    with pytest.raises(ValueError, match="video"):
        asyncio.run(preprocess(StagePayload("request", request, {})))


class ScriptedSession:
    """Plays back one outcome per step; an exception stands for the engine raising it."""

    def __init__(
        self, outcomes: list[TextEvent | EndEvent | Exception], *, is_exhausted: bool
    ) -> None:
        self.outcomes = outcomes
        self.is_exhausted = is_exhausted

    def step(self) -> TextEvent | EndEvent:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        else:
            return outcome


class EngineHarness:
    """The real engine scheduler and session bookkeeping around scripted sessions."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scripts: list[list[TextEvent | EndEvent | Exception]],
        *,
        is_exhausted: bool = False,
    ) -> None:
        tokenizer = SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens: f"decoded{token_ids}"
        )
        runtime = SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer))
        self.started = 0
        self.on_start = None
        self.scripts = scripts

        def start_session(unused_runtime: object, inputs: object) -> ScriptedSession:
            session = ScriptedSession(
                self.scripts[self.started], is_exhausted=is_exhausted
            )
            self.started += 1
            if self.on_start is not None:
                self.on_start()
            else:
                pass
            return session

        monkeypatch.setattr(stages, "load_lumina_runtime", lambda *_: runtime)
        monkeypatch.setattr(stages, "start_session", start_session)
        self.scheduler = stages.create_engine_executor("model", device="cpu")
        self.loop = asyncio.new_event_loop()

    def dispatch(self, request_id: str = "request") -> dict[str, object] | None:
        """Run one event dispatch; the result is None when the scheduler emits nothing."""
        payload = StagePayload(request_id, OmniRequest({}, {}, {}), {})
        message = IncomingMessage(
            request_id=request_id, type="new_request", data=payload
        )
        self.scheduler.run_single(message, self.loop)
        if self.scheduler.outbox.empty():
            return None
        else:
            return self.scheduler.outbox.get_nowait().data.data


def test_events_of_one_request_share_a_session_until_it_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = EngineHarness(
        monkeypatch,
        [
            [TextEvent([1]), IncompleteInterleavedGenerationError("budget")],
            [TextEvent([2])],
        ],
    )
    assert harness.dispatch()["output"] == {"kind": "text", "text": "decoded[1]"}
    assert harness.dispatch() == {"output": None, "finish_reason": "length"}
    assert harness.started == 1
    harness.dispatch()
    assert harness.started == 2


def test_truncated_text_still_reaches_the_client_and_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = IncompleteTextGenerationError("budget", token_ids=[5, 6])
    harness = EngineHarness(monkeypatch, [[error]])
    assert harness.dispatch() == {
        "output": {"kind": "text", "text": "decoded[5, 6]"},
        "finish_reason": "length",
    }


def test_the_model_closing_its_turn_stops_and_running_out_of_decisions_is_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausting the decisions raises nothing, so it must not read as a closed turn."""
    closed = EngineHarness(monkeypatch, [[EndEvent()]], is_exhausted=True)
    assert closed.dispatch() == {"output": None, "finish_reason": "stop"}
    exhausted = EngineHarness(monkeypatch, [[TextEvent([1])]], is_exhausted=True)
    assert exhausted.dispatch() == {
        "output": {"kind": "text", "text": "decoded[1]"},
        "finish_reason": "length",
    }


def test_abort_landing_while_a_session_starts_still_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session is stored only after the abort, so one cleanup pass would leak it."""
    harness = EngineHarness(monkeypatch, [[TextEvent([1])], [TextEvent([2])]])

    def abort_once() -> None:
        harness.on_start = None
        harness.scheduler.abort("request")

    harness.on_start = abort_once
    assert harness.dispatch() is None
    harness.dispatch()
    assert harness.started == 2


def test_a_failed_event_releases_its_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = EngineHarness(
        monkeypatch, [[RuntimeError("out of memory")], [TextEvent([2])]]
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        harness.dispatch()
    harness.dispatch()
    assert harness.started == 2
