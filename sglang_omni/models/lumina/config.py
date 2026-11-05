# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration of Lumina interleaved generation."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import FactoryArgs, StageConfig
from sglang_omni.pipeline.umm import (
    CONTROLLER_FACTORY_PATH,
    OUTPUT_FACTORY_PATH,
    ROUTE_FN_PATH,
    STREAM_DONE_FN_PATH,
    UMMPipelineConfig,
)

PACKAGE = "sglang_omni.models.lumina"
PREPROCESSING_STAGE = "preprocessing"
ENGINE_STAGE = "engine"


class LuminaPipelineConfig(UMMPipelineConfig):
    """Text and images in one answer, on the shared UMM controller.

    Phase prepare resolves media on the preprocessing stage. Phase event runs on
    the engine, whose process owns the model and its process-wide numerics.
    """

    architecture: ClassVar[str] = "LuminaForConditionalGeneration"

    model_path: str
    entry_stage: str = "orchestrator"
    stages: list[StageConfig] = [
        StageConfig(
            name="orchestrator",
            process="orchestrator",
            factory_path=CONTROLLER_FACTORY_PATH,
            factory=FactoryArgs(
                adapter=f"{PACKAGE}.umm.LuminaUMMAdapter",
                phases={"prepare": PREPROCESSING_STAGE, "event": ENGINE_STAGE},
                # Preparation and nine events: the default max_events allows nine decisions.
                max_dispatches=10,
                max_outputs=9,
                # Bounds orchestrator memory. The engine queues what it cannot run yet.
                max_active=32,
                # Four PNG data URLs at the largest edge, about 3 MiB each, and their text.
                max_retained_bytes=64 * 1024 * 1024,
                timeout_s=600.0,
            ),
            next=[PREPROCESSING_STAGE, ENGINE_STAGE, "output"],
            route_fn=ROUTE_FN_PATH,
            stream_to=["output"],
            stream_done_to_fn=STREAM_DONE_FN_PATH,
        ),
        StageConfig(
            name="output",
            process="orchestrator",
            factory_path=OUTPUT_FACTORY_PATH,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
        StageConfig(
            name=PREPROCESSING_STAGE,
            process=PREPROCESSING_STAGE,
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next="orchestrator",
        ),
        StageConfig(
            name=ENGINE_STAGE,
            process=ENGINE_STAGE,
            factory_path=f"{PACKAGE}.stages.create_engine_executor",
            factory=FactoryArgs(
                # About 33% faster but not token-equal to the eager decoder, and it needs CUDA_HOME.
                compile_decoder=False,
            ),
            gpu=0,
            next="orchestrator",
        ),
    ]


EntryClass = LuminaPipelineConfig
