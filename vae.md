# VAE Training Notes

This file collects practical run suggestions for the VAE decoder fine-tuning
path in ACE-Step.

This guidance assumes:

- About 20 GB of VRAM available.
- Raw audio clips already under 50 seconds.
- You want to keep each file intact rather than split it into smaller chunks.
- The audio is meant to be used as-is for looping.

The loader processes each file as one training sample and pads only at batch
time if a batch contains different-length clips.

## Recommended Baseline

If you want one configuration to begin with, use this:

| Setting | Recommended value | Why |
| --- | --- | --- |
| `val_split` | `0.05` | Small validation slice without wasting much data. |
| `learning_rate` | `5e-5` | Safer starting point for whole-file training. |
| `batch_size` | `1` | Conservative default for 20 GB VRAM with variable clip lengths. |
| `gradient_accumulation` | `2` | Gives you a larger effective batch without extra memory pressure. |
| `max_epochs` | `25` to `50` | Good first training window for a 2h to 8h dataset. |
| `save_every_n_epochs` | `5` | Frequent enough to keep recovery points. |
| `l1_loss_weight` | `1.0` | Preserves waveform fidelity. |
| `stft_loss_weight` | `1.0` | Preserves spectral detail. |
| `freeze_encoder` | `true` | Keeps the latent space compatible with inference. |

## Recommended Run Profiles

### Around 2 Hours Of Audio

Use this when you have a smaller but clean dataset and want to avoid overfitting.

- `batch_size`: `1`
- `gradient_accumulation`: `2`
- `learning_rate`: `5e-5`
- `max_epochs`: `30` to `60`
- `save_every_n_epochs`: `5`
- `val_split`: `0.05` to `0.10`

### Around 8 Hours Of Audio

Use this when you have enough material to train a more general decoder.

- `batch_size`: `1`
- `gradient_accumulation`: `2` to `4`
- `learning_rate`: `5e-5`
- `max_epochs`: `15` to `35`
- `save_every_n_epochs`: `5`
- `val_split`: `0.03` to `0.05`

More data means each epoch already covers a lot of audio, so you usually need
fewer epochs than with a small dataset.

## Dataset Suggestions

- Use clean, consistent audio that matches the target style.
- Prefer clean whole clips over heavily edited snippets.
- Avoid clipping, silence-heavy files, and aggressive transcoding artifacts.
- Keep the source audio close to the inference domain.

## How To Adjust A Run

- If the loss is noisy or unstable, lower `learning_rate` to `2.5e-5` or
  `5e-5`.
- If you run out of memory, keep `batch_size = 1` and reduce
  `gradient_accumulation` before changing the data length or model settings.
- If training is too slow, raise `gradient_accumulation` only if memory allows.
- If validation loss rises while training loss keeps dropping, stop earlier or
  lower the learning rate.
- If the output sounds dull, raise `stft_loss_weight` to `1.5`.
- If the output sounds sharp or noisy, keep `stft_loss_weight` near `1.0` and
  lower the learning rate first.

## Good Run Checklist

- Use `freeze_encoder = true` unless you intentionally want to change the
  latent space.
- Keep the loader in whole-file mode so the clips stay intact.
- Keep `val_split` between `0.03` and `0.10`.
- Save every few epochs so you have fallback checkpoints.
- Watch for `NaN` or flat loss curves and stop early if they appear.
- Resume from `checkpoints/best` when validation is enabled and the run looks
  promising.

## Checkpoint Usage

- `output_dir/checkpoints/best` is the best checkpoint when validation is
  enabled.
- `output_dir/checkpoints/epoch_*` holds periodic snapshots.
- `output_dir/final` is the final decoder export target.
- The UI export button copies the final decoder directory to your chosen export
  path.
