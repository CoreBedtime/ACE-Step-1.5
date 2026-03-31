"""VAE decoder training run-wiring helpers.

Registers all Gradio event handlers for the VAE fine-tuning tab,
following the same pattern as ``training_run_wiring.py`` and
``training_lokr_wiring.py``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from loguru import logger

from acestep.api.train_api_runtime import RuntimeComponentManager
from acestep.ui.gradio.events.training.vae_training import (
    export_vae_decoder,
    scan_vae_dataset,
    start_vae_training,
    stop_vae_training,
)
from .context import TrainingWiringContext


def _build_vae_training_wrapper(
    dit_handler: Any,
    llm_handler: Any,
    normalize_training_state: Callable[[Any], dict[str, bool]],
) -> Callable[..., Iterator[tuple[Any, Any, Any, dict[str, bool]]]]:
    """Build a closure that streams VAE training progress.

    Args:
        dit_handler: Initialised DiT handler passed to the trainer.
        llm_handler: Optional LLM handler to temporarily unload during VAE training.
        normalize_training_state: Callable that coerces any state value to
            a valid ``dict[str, bool]`` mapping.

    Returns:
        A generator function compatible with Gradio's streaming ``click`` API.
    """

    def vae_training_wrapper(
        audio_dir: Any,
        val_split: Any,
        learning_rate: Any,
        train_epochs: Any,
        batch_size: Any,
        gradient_accumulation: Any,
        save_every_n_epochs: Any,
        l1_loss_weight: Any,
        stft_loss_weight: Any,
        training_seed: Any,
        output_dir: Any,
        freeze_encoder: Any,
        resume_checkpoint_dir: Any,
        training_state: Any,
    ) -> Iterator[tuple[Any, Any, Any, dict[str, bool]]]:
        """Stream VAE training progress; normalise failure outputs for the UI."""

        state = normalize_training_state(training_state)
        component_manager: RuntimeComponentManager | None = None
        try:
            component_manager = RuntimeComponentManager(
                handler=dit_handler,
                llm=llm_handler,
                app_state=None,
            )
            component_manager.offload_model_to_cpu()
            component_manager.offload_text_encoder_to_cpu()
            component_manager.unload_llm()
            yield from start_vae_training(
                audio_dir=audio_dir,
                dit_handler=dit_handler,
                val_split=val_split,
                learning_rate=learning_rate,
                train_epochs=train_epochs,
                batch_size=batch_size,
                gradient_accumulation=gradient_accumulation,
                save_every_n_epochs=save_every_n_epochs,
                l1_loss_weight=l1_loss_weight,
                stft_loss_weight=stft_loss_weight,
                training_seed=training_seed,
                output_dir=output_dir,
                freeze_encoder=freeze_encoder,
                resume_checkpoint_dir=resume_checkpoint_dir,
                training_state=state,
            )
        except Exception as exc:  # pragma: no cover — defensive UI wrapper
            logger.exception("VAE training wrapper error")
            yield f"\u274c Error: {exc!s}", f"{exc!s}", None, state
        finally:
            if component_manager is not None:
                component_manager.restore()

    return vae_training_wrapper


def register_vae_training_handlers(
    context: TrainingWiringContext,
    *,
    normalize_training_state: Callable[[Any], dict[str, bool]],
) -> None:
    """Register all Gradio event handlers for the VAE fine-tuning tab.

    Args:
        context: Shared wiring context carrying the component map and
            the DiT handler reference.
        normalize_training_state: Callable (imported from
            ``training_run_wiring``) that guarantees the training state
            is a valid mutable dict.
    """
    s = context.training_section
    vae_wrapper = _build_vae_training_wrapper(
        context.dit_handler,
        context.llm_handler,
        normalize_training_state,
    )

    # -- Dataset scan --
    s["vae_load_dataset_btn"].click(
        fn=scan_vae_dataset,
        inputs=[s["vae_audio_dir"]],
        outputs=[s["vae_dataset_info"]],
    )

    # -- Start training (streaming) --
    s["vae_start_training_btn"].click(
        fn=vae_wrapper,
        inputs=[
            s["vae_audio_dir"],
            s["vae_val_split"],
            s["vae_learning_rate"],
            s["vae_train_epochs"],
            s["vae_batch_size"],
            s["vae_gradient_accumulation"],
            s["vae_save_every_n_epochs"],
            s["vae_l1_loss_weight"],
            s["vae_stft_loss_weight"],
            s["vae_training_seed"],
            s["vae_output_dir"],
            s["vae_freeze_encoder"],
            s["vae_resume_checkpoint_dir"],
            s["training_state"],
        ],
        outputs=[
            s["vae_training_progress"],
            s["vae_training_log"],
            s["vae_training_loss_plot"],
            s["training_state"],
        ],
    )

    # -- Stop --
    s["vae_stop_training_btn"].click(
        fn=stop_vae_training,
        inputs=[s["training_state"]],
        outputs=[
            s["vae_training_progress"],
            s["training_state"],
        ],
    )

    # -- Export --
    s["vae_export_btn"].click(
        fn=export_vae_decoder,
        inputs=[
            s["vae_export_path"],
            s["vae_output_dir"],
        ],
        outputs=[s["vae_export_status"]],
    )
