"""VAE decoder fine-tuning tab for the Gradio training interface.

Provides dataset directory selection, training hyperparameter controls,
live loss plot, and checkpoint export — mirroring the LoRA training tab
structure.
"""

from __future__ import annotations

import gradio as gr

from acestep.ui.gradio.i18n import t


def build_vae_training_controls() -> dict[str, object]:
    """Build the VAE decoder fine-tuning controls.

    Returns:
        Mapping of component keys to Gradio components for event wiring.
    """
    gr.HTML(
        """
        <div style="margin-bottom: 12px;">
          <h3>VAE Decoder Fine-Tuning</h3>
          <p>
            Fine-tune the <b>AutoencoderOobleck</b> decoder on your audio dataset.
            The encoder is kept frozen; only decoder parameters are trained using a
            weighted L1 + multi-scale STFT reconstruction loss.
          </p>
          <p style="color: #888; font-size: 0.9em;">
            Requires raw audio files (WAV/FLAC/MP3/OGG) — no preprocessing needed.
            Each file is used as a whole sample, without chunking or windowing.
          </p>
        </div>
        """
    )

    tab_controls: dict[str, object] = {}
    tab_controls.update(_build_vae_dataset_controls())
    tab_controls.update(_build_vae_hyperparameter_controls())
    tab_controls.update(_build_vae_run_controls())
    tab_controls.update(_build_vae_export_controls())

    return tab_controls


def create_training_vae_tab() -> dict[str, object]:
    """Create the VAE decoder fine-tuning tab.

    Returns:
        Mapping of component keys to Gradio components for event wiring.
    """
    with gr.Tab(t("vae.tab_title")):
        return build_vae_training_controls()


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------


def _build_vae_dataset_controls() -> dict[str, object]:
    """Render audio-directory selector and dataset-info display.

    Returns:
        Dict of component keys for event wiring.
    """
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<h4>Dataset</h4>")
            vae_audio_dir = gr.Textbox(
                label="Audio Directory",
                placeholder="./datasets/my_audio",
                value="./datasets/my_audio",
                info=(
                    "Directory tree of raw audio files (WAV, FLAC, MP3, OGG, AAC, M4A). "
                    "Recursive scan — each file is loaded as one training sample. "
                    "Recommended: clean audio that matches the target timbre/style."
                ),
                elem_classes=["has-info-container"],
            )
            vae_load_dataset_btn = gr.Button("Scan Dataset", variant="secondary")
            vae_dataset_info = gr.Textbox(
                label="Dataset Info",
                interactive=False,
                lines=3,
            )
        with gr.Column(scale=1):
            gr.HTML("<h4>Validation</h4>")
            vae_val_split = gr.Slider(
                minimum=0.0,
                maximum=0.3,
                step=0.01,
                value=0.05,
                label="Validation Split",
                info=(
                    "Fraction of files held out for validation loss tracking. "
                    "0 = no validation."
                ),
                elem_classes=["has-info-container"],
            )
            gr.HTML(
                """
                <p style="color: #888; font-size: 0.9em;">
                  No chunking is applied. Files are loaded and padded only at batch
                  time if needed.
                </p>
                """
            )

    return {
        "vae_audio_dir": vae_audio_dir,
        "vae_load_dataset_btn": vae_load_dataset_btn,
        "vae_dataset_info": vae_dataset_info,
        "vae_val_split": vae_val_split,
    }


def _build_vae_hyperparameter_controls() -> dict[str, object]:
    """Render training hyperparameter controls.

    Returns:
        Dict of component keys for event wiring.
    """
    gr.HTML("<hr><h4>Training Hyperparameters</h4>")

    with gr.Row():
        vae_learning_rate = gr.Number(
            label="Learning Rate",
            value=1e-4,
            info=(
                "AdamW learning rate. 1e-4 is a safe default. "
                "Lower (5e-5) for fine-grained style transfers; "
                "higher (2e-4) for large datasets."
            ),
            elem_classes=["has-info-container"],
        )
        vae_train_epochs = gr.Slider(
            minimum=1,
            maximum=500,
            step=1,
            value=50,
            label="Epochs",
        )
        vae_batch_size = gr.Slider(
            minimum=1,
            maximum=8,
            step=1,
            value=1,
            label="Batch Size",
            info="Samples per gradient step. Increase if VRAM allows.",
            elem_classes=["has-info-container"],
        )
        vae_gradient_accumulation = gr.Slider(
            minimum=1,
            maximum=16,
            step=1,
            value=4,
            label="Gradient Accumulation",
            info=(
                "Effective batch = batch_size × accumulation. "
                "Higher values stabilise training on small batches."
            ),
            elem_classes=["has-info-container"],
        )

    with gr.Row():
        vae_save_every_n_epochs = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=5,
            label="Save Every N Epochs",
        )
        vae_l1_loss_weight = gr.Number(
            label="L1 Loss Weight",
            value=1.0,
            info="Weight for waveform L1 reconstruction loss.",
            elem_classes=["has-info-container"],
        )
        vae_stft_loss_weight = gr.Number(
            label="STFT Loss Weight",
            value=1.0,
            info=(
                "Weight for multi-scale STFT spectral loss. "
                "Increasing this emphasises frequency accuracy."
            ),
            elem_classes=["has-info-container"],
        )
        vae_training_seed = gr.Number(
            label="Seed",
            value=42,
            precision=0,
        )

    with gr.Row():
        vae_output_dir = gr.Textbox(
            label="Output Directory",
            value="./vae_output",
            placeholder="./vae_output",
            info="Checkpoints and final weights are saved here.",
            elem_classes=["has-info-container"],
        )
        vae_freeze_encoder = gr.Checkbox(
            label="Freeze Encoder",
            value=True,
            info=(
                "Recommended: keep encoder frozen so the latent space stays "
                "compatible with the DiT decoder used during inference."
            ),
            elem_classes=["has-info-container"],
        )

    with gr.Row():
        vae_resume_checkpoint_dir = gr.Textbox(
            label="Resume Checkpoint",
            placeholder="./vae_output/checkpoints/epoch_10",
            info="Directory of a saved VAE checkpoint to resume training from.",
            elem_classes=["has-info-container"],
        )

    return {
        "vae_learning_rate": vae_learning_rate,
        "vae_train_epochs": vae_train_epochs,
        "vae_batch_size": vae_batch_size,
        "vae_gradient_accumulation": vae_gradient_accumulation,
        "vae_save_every_n_epochs": vae_save_every_n_epochs,
        "vae_l1_loss_weight": vae_l1_loss_weight,
        "vae_stft_loss_weight": vae_stft_loss_weight,
        "vae_training_seed": vae_training_seed,
        "vae_output_dir": vae_output_dir,
        "vae_freeze_encoder": vae_freeze_encoder,
        "vae_resume_checkpoint_dir": vae_resume_checkpoint_dir,
    }


def _build_vae_run_controls() -> dict[str, object]:
    """Render start/stop buttons, progress textbox, and loss plot.

    Returns:
        Dict of component keys for event wiring.
    """
    gr.HTML("<hr>")

    with gr.Row():
        with gr.Column(scale=1):
            vae_start_training_btn = gr.Button(
                "Start VAE Training",
                variant="primary",
                size="lg",
            )
        with gr.Column(scale=1):
            vae_stop_training_btn = gr.Button(
                "Stop Training",
                variant="stop",
                size="lg",
            )

    vae_training_progress = gr.Textbox(
        label="Progress",
        interactive=False,
        lines=2,
    )

    with gr.Row():
        vae_training_log = gr.Textbox(
            label="Training Log",
            interactive=False,
            lines=10,
            max_lines=15,
            scale=1,
        )
        vae_training_loss_plot = gr.Plot(
            label="Training Loss",
            scale=1,
        )

    return {
        "vae_start_training_btn": vae_start_training_btn,
        "vae_stop_training_btn": vae_stop_training_btn,
        "vae_training_progress": vae_training_progress,
        "vae_training_log": vae_training_log,
        "vae_training_loss_plot": vae_training_loss_plot,
    }


def _build_vae_export_controls() -> dict[str, object]:
    """Render checkpoint export controls.

    Returns:
        Dict of component keys for event wiring.
    """
    gr.HTML("<hr><h4>Export</h4>")

    with gr.Row():
        vae_export_path = gr.Textbox(
            label="Export Path",
            value="./vae_output/final_vae_decoder",
            placeholder="./vae_output/my_vae_decoder",
            info="Directory where the final VAE decoder weights will be copied.",
            elem_classes=["has-info-container"],
        )
        vae_export_btn = gr.Button("Export VAE Decoder", variant="secondary")

    vae_export_status = gr.Textbox(
        label="Export Status",
        interactive=False,
    )

    return {
        "vae_export_path": vae_export_path,
        "vae_export_btn": vae_export_btn,
        "vae_export_status": vae_export_status,
    }
