# vedaz-qwen-finetune

Fine-tuned **Qwen2.5-0.5B-Instruct** on Vedaz's AI Vedic astrology chat data.

**Assessment for:** Vedaz — AI Internship (Internshala)

---

## What's in this repo

| File | Description |
|---|---|
| `finetune_qwen.py` | QLoRA fine-tuning script (runs on CPU, no GPU needed) |
| `vllm_hosting_guide.md` | Full write-up: hosting the model on VPS using vLLM |

## Fine-tuned Model Weights

The trained LoRA adapter is live on Hugging Face Hub:

👉 **[rohit1425/vedaz-qwen2.5-0.5B-finetune](https://huggingface.co/rohit1425/vedaz-qwen2.5-0.5B-finetune)**

## Dataset

Vedaz Vedic astrology chat data — 55 multi-turn conversations in Hindi, English & Hinglish.

- Format: OpenAI `messages` (system / user / assistant)
- Topics: marriage, career, visa, health, relationships, mental health crisis handling

## Training Details

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA (PEFT) — CPU fine-tuning |
| LoRA rank | 8, alpha 16 |
| Trainable params | 4.4M / 498M (0.88%) |
| Epochs | 1 |
| Train loss | 2.557 → 1.225 |
| Eval loss | 3.232 |
| Training time | ~15 min (CPU) |

## How to Run

```bash
pip install transformers trl peft datasets torch
python finetune_qwen.py
```

## vLLM Deployment

See [`vllm_hosting_guide.md`](./vllm_hosting_guide.md) for the full step-by-step guide on:
- VPS selection & GPU specs
- CUDA + vLLM installation
- Serving the model with OpenAI-compatible API
- Systemd + Nginx + SSL production setup
