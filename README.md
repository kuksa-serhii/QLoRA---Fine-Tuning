# QLoRA Fine-Tuning Pipeline

This project contains a pipeline for fine-tuning large language models using QLoRA (Quantized Low-Rank Adaptation) with support for two methods:
- **SFT (Supervised Fine-Tuning)** - supervised training
- **DPO (Direct Preference Optimization)** - preference-based optimization

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended A100 40GB or equivalent)
- **RAM**: Minimum 32GB system memory
- **Disk**: 50GB+ free space (for models and datasets)

### Software
- Python 3.10+
- CUDA 11.8+ or 12.1+
- Linux/Windows with GPU support

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd QLoRA---Fine-Tuning
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Linux/Mac)
source venv/bin/activate
```

Or using conda:
```bash
conda create -n qlora python=3.10
conda activate qlora
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA (if needed)
If you don't have PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Verify Installation
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

## 📁 Project Structure

```
QLoRA---Fine-Tuning/
│
├── ft_pipeline/              # Main pipeline
│   ├── config.py            # Configurations (FTConfig, DPOCfg)
│   ├── model.py             # Model and tokenizer loading
│   ├── data.py              # Dataset loading and processing
│   ├── trainer.py           # Trainers (SFT, DPO)
│   ├── run_sft.py           # SFT execution
│   ├── run_dpo.py           # DPO execution
│   ├── callbacks.py         # Monitoring callbacks
│   ├── ab_eval.py           # A/B testing
│   ├── infer.py             # Inference
│   ├── logger.py            # Logging
│   └── env.py               # Environment setup
│
├── QLoRA - SFT - Supervised Fine-Tuning.ipynb    # Notebook for SFT
├── QLoRA - DPO - Direct Preference Optimization.ipynb  # Notebook for DPO
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📊 Dataset Preparation

### SFT Format
Create `sft_train.jsonl` and `sft_val.jsonl` files in the `ft_datasets/` folder:

```jsonl
{"messages": [{"role": "user", "content": "User question"}], "target": {"content": "Assistant answer"}}
{"messages": [{"role": "user", "content": "Another question"}], "target": {"content": "Another answer"}}
```

### DPO Format
Create `dpo_train.jsonl` and `dpo_val.jsonl` files:

```jsonl
{"messages": [{"role": "user", "content": "Question"}], "chosen": "Good answer", "rejected": "Bad answer"}
{"messages": [{"role": "user", "content": "Another question"}], "chosen": "Better answer", "rejected": "Worse answer"}
```

## 🎯 Usage

### Option 1: Jupyter Notebooks (Recommended)

#### SFT Training
1. Open `QLoRA - SFT - Supervised Fine-Tuning.ipynb`
2. Configure parameters in Cell 2 (FTConfig):
   - `model_id`: path to base model
   - `train_jsonl`, `val_jsonl`: paths to datasets
   - `out_dir`: folder for saving results
   - Other hyperparameters (learning_rate, batch_size, etc.)
3. Run all cells sequentially

#### DPO Training
1. Open `QLoRA - DPO - Direct Preference Optimization.ipynb`
2. Configure DPOCfg:
   - `model_id`: base model
   - `sft_adapter_dir`: path to LoRA adapter after SFT (or `None` for DPO from base model)
   - `dpo_train_jsonl`, `dpo_val_jsonl`: datasets
3. Run all cells

### Option 2: Python Scripts

```python
from ft_pipeline.config import FTConfig
from ft_pipeline.run_sft import run_finetune

# Configuration setup
cfg = FTConfig(
    model_id="your-model-path",
    train_jsonl="ft_datasets/sft_train.jsonl",
    val_jsonl="ft_datasets/sft_val.jsonl",
    out_dir="outputs/sft",
    max_seq_len=2048,
    num_train_epochs=2,
)

# Start training
artifacts = run_finetune(cfg)
```

## ⚙️ Key Parameters

### SFT (Supervised Fine-Tuning)
- `max_seq_len`: Maximum sequence length (1024-8192)
- `per_device_train_batch_size`: Batch size per GPU (usually 1 for large models)
- `gradient_accumulation_steps`: Gradient accumulation steps (8-16)
- `learning_rate`: Learning rate (0.00001 - 0.0001)
- `lora_r`: LoRA rank (8, 16, 32)
- `lora_alpha`: LoRA alpha (16, 32, 64)

### DPO (Direct Preference Optimization)
- `beta`: DPO beta parameter (0.01 - 0.1) - controls preference optimization strength
- `sft_adapter_dir`: Path to pre-trained SFT adapter
- All other parameters similar to SFT

## 📈 Monitoring

Training logs are saved to:
- `{out_dir}/ft_run_sft.log` for SFT
- `{out_dir}/ft_run_dpo.log` for DPO

Checkpoints are saved to:
- `{out_dir}/checkpoint-{step}/`
- `{out_dir}/lora_adapter/` - final adapter

## 🔍 A/B Testing

The pipeline automatically performs A/B comparison before and after training on validation examples. Results are saved in the corresponding output folder.

## 💡 Tips and Best Practices

1. **GPU Memory**:
   - For 7B-13B models: requires 16-24GB VRAM
   - For 30B+ models: requires 40GB+ VRAM
   - Use `gradient_checkpointing` to save memory

2. **Hyperparameters**:
   - Start with small learning rate (0.00001-0.0001)
   - Use cosine scheduler
   - LoRA rank 16 - good balance for most tasks

3. **Datasets**:
   - Minimum 100-500 examples for SFT
   - For DPO, recommended 1000+ pairs (chosen/rejected)

4. **Validation**:
   - Regularly check metrics on validation set
   - Use early stopping to prevent overfitting

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce `max_seq_len`
- Set `per_device_train_batch_size=1`
- Increase `gradient_accumulation_steps`
- Enable gradient checkpointing

### Slow Training
- Check if GPU is being used: `torch.cuda.is_available()`
- Enable `use_bf16=True` for A100
- Check attn_implementation="sdpa"

### Import Errors
- Ensure all packages are installed: `pip install -r requirements.txt`
- Update transformers: `pip install --upgrade transformers`

## 📝 License

[Your License]

## 🤝 Contributing

[Instructions for contributors]

## 📧 Contact

[Your contacts]

---

**Note**: This project is optimized for NVIDIA A100 40GB, but can work on other GPUs with appropriate parameter adjustments.
