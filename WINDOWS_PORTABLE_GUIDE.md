# Windows Portable Package - Quick Start Guide

This guide is specifically for users of the **Windows Portable Package** version of ACE-Step 1.5.

## 🚀 Quick Start

1. **Extract** the portable package to a folder
2. **Double-click** `start_gradio_ui.bat` to launch the web interface
3. **Open** your browser to `http://127.0.0.1:7860`

## ⚙️ Configuration

### Service Configuration Tab is Missing?

The Windows Portable version **pre-initializes** the service on startup for faster loading. This hides the Service Configuration tab in the web interface.

**You have two options:**

### Option 1: Configure via `start_gradio_ui.bat` (Recommended)

Edit `start_gradio_ui.bat` in a text editor (Notepad, VS Code, etc.) to configure service settings:

```batch
REM ==================== Service Configuration ====================
REM Flash Attention settings - Disable if you have GPU errors
set USE_FLASH_ATTENTION=--use_flash_attention false

REM Model selection
set CONFIG_PATH=--config_path acestep-v15-turbo
set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-0.6B

REM Performance settings
set OFFLOAD_TO_CPU=--offload_to_cpu true

REM LLM initialization (auto, true, or false)
set INIT_LLM=--init_llm auto
```

**Save the file** and **restart** the application by running `start_gradio_ui.bat` again.

### Option 2: Show Service Configuration Tab in UI

To access the full Service Configuration tab in the web interface:

1. **Edit** `start_gradio_ui.bat`
2. **Find** this line (around line 44):
   ```batch
   set INIT_SERVICE=--init_service true
   ```
3. **Comment it out** by adding `REM` at the start:
   ```batch
   REM set INIT_SERVICE=--init_service true
   ```
   Or change it to:
   ```batch
   set INIT_SERVICE=
   ```
4. **Save and restart** the application

The Service Configuration tab will now be visible in the web interface, allowing you to configure all settings there.

## 🔧 Common Issues

### Flash Attention Errors

**Symptoms:**
- Application crashes during initialization
- CUDA errors or "flash_attn" error messages
- GPU compatibility issues

**Solution:**

Edit `start_gradio_ui.bat` and add this line:
```batch
set USE_FLASH_ATTENTION=--use_flash_attention false
```

Flash Attention requires specific GPU architectures (NVIDIA Ampere or newer). Disabling it uses standard attention, which is slower but more compatible.

### Out of Memory Errors

**For GPUs with limited VRAM (<8GB):**

1. Enable CPU offload:
   ```batch
   set OFFLOAD_TO_CPU=--offload_to_cpu true
   ```

2. Disable LLM (Language Model):
   ```batch
   set INIT_LLM=--init_llm false
   ```

3. Use the smaller turbo model:
   ```batch
   set CONFIG_PATH=--config_path acestep-v15-turbo
   ```

### Slow Generation

**If generation is taking too long:**

1. Enable Flash Attention (if your GPU supports it):
   ```batch
   set USE_FLASH_ATTENTION=--use_flash_attention true
   ```

2. Use the smaller LM model:
   ```batch
   set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-0.6B
   ```

3. Disable CPU offload (if you have enough VRAM):
   ```batch
   REM set OFFLOAD_TO_CPU=--offload_to_cpu true
   ```

## 📝 All Configurable Settings

Here are all the settings you can configure in `start_gradio_ui.bat`:

| Setting | Description | Example |
|---------|-------------|---------|
| `PORT` | Web server port | `set PORT=7860` |
| `SERVER_NAME` | Server address | `set SERVER_NAME=127.0.0.1` |
| `LANGUAGE` | UI language (en, zh, ja, he) | `set LANGUAGE=en` |
| `CONFIG_PATH` | DiT model | `set CONFIG_PATH=--config_path acestep-v15-turbo` |
| `LM_MODEL_PATH` | Language model | `set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-0.6B` |
| `INIT_LLM` | Enable LLM (auto, true, false) | `set INIT_LLM=--init_llm auto` |
| `USE_FLASH_ATTENTION` | Flash attention | `set USE_FLASH_ATTENTION=--use_flash_attention false` |
| `OFFLOAD_TO_CPU` | CPU offload | `set OFFLOAD_TO_CPU=--offload_to_cpu true` |
| `INIT_SERVICE` | Pre-initialize service | `set INIT_SERVICE=--init_service true` |
| `CHECK_UPDATE` | Check for updates | `set CHECK_UPDATE=true` |
| `SHARE` | Create public link | `set SHARE=--share` |

## 📚 Documentation

For more detailed information, see:

- **[Gradio User Guide](./docs/en/GRADIO_GUIDE.md)** - Complete web interface documentation
- **[GPU Compatibility](./docs/en/GPU_COMPATIBILITY.md)** - GPU requirements and troubleshooting
- **[CLI Guide](./docs/en/CLI.md)** - Command-line interface documentation
- **[Tutorial](./docs/en/Tutorial.md)** - Comprehensive usage guide

## 🆘 Getting Help

If you encounter issues:

1. Check the [Troubleshooting section](./docs/en/GPU_COMPATIBILITY.md#troubleshooting) in the GPU Compatibility Guide
2. Review the [Gradio Guide - Service Configuration](./docs/en/GRADIO_GUIDE.md#service-configuration)
3. Open an issue on [GitHub](https://github.com/ace-step/ACE-Step-1.5/issues)

## 🔄 Updates

To check for updates:

1. Set `CHECK_UPDATE=true` in `start_gradio_ui.bat`
2. Restart the application
3. Updates will be checked automatically on startup

Or manually run `check_update.bat` in the ACE-Step directory.
