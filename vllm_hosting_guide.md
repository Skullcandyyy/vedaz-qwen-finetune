# Hosting a Fine-Tuned Qwen Model on VPS Using vLLM

**Model:** Vedaz-Astro (Qwen2.5-7B fine-tuned)  
**Serving stack:** vLLM · OpenAI-compatible REST API  
**OS:** Ubuntu 22.04 LTS

---

## Table of Contents

1. [Why vLLM?](#1-why-vllm)
2. [VPS Selection & Recommended Specs](#2-vps-selection--recommended-specs)
3. [Server Setup](#3-server-setup)
4. [Install CUDA & Python Environment](#4-install-cuda--python-environment)
5. [Install vLLM](#5-install-vllm)
6. [Upload the Fine-Tuned Model](#6-upload-the-fine-tuned-model)
7. [Serve the Model with vLLM](#7-serve-the-model-with-vllm)
8. [Test the API](#8-test-the-api)
9. [Production Setup (Systemd + Nginx + SSL)](#9-production-setup-systemd--nginx--ssl)
10. [Monitoring & Scaling](#10-monitoring--scaling)
11. [Cost Estimates](#11-cost-estimates)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Why vLLM?

vLLM is the industry-standard engine for serving large language models in production:

| Feature | Benefit |
|---|---|
| **PagedAttention** | 2–4× higher throughput by eliminating KV-cache fragmentation |
| **Continuous batching** | Serves multiple requests simultaneously without padding waste |
| **OpenAI-compatible API** | Drop-in replacement for `gpt-3.5-turbo` — no client code changes |
| **Streaming** | Server-sent events (SSE) for real-time token streaming |
| **AWQ / GPTQ / FP8 support** | Multiple quantization formats to fit different VRAM budgets |
| **Multi-GPU tensor parallelism** | Scale across 2–8 GPUs with a single flag |

---

## 2. VPS Selection & Recommended Specs

### Minimum (for Qwen2.5-7B in fp16)

| Resource | Requirement |
|---|---|
| GPU | 1× NVIDIA A10G (24 GB) or RTX 3090 (24 GB) |
| CPU | 8 vCPU |
| RAM | 32 GB |
| Disk | 100 GB NVMe SSD |
| OS | Ubuntu 22.04 LTS |

### Recommended (for production with good throughput)

| Resource | Recommendation |
|---|---|
| GPU | 1× A100 (40 GB) or 2× A10G |
| CPU | 16 vCPU |
| RAM | 64 GB |
| Disk | 200 GB NVMe SSD |

### GPU Cloud Provider Options

| Provider | GPU | Approx. Cost/hr | Notes |
|---|---|---|---|
| **RunPod** | A10G (24GB) | ~$0.37/hr | Best value, fast setup |
| **Lambda Labs** | A10 (24GB) | ~$0.60/hr | Stable, good SLA |
| **Vast.ai** | RTX 3090 (24GB) | ~$0.20–$0.40/hr | Cheapest, community GPUs |
| **AWS EC2** | g5.xlarge (A10G) | ~$1.00/hr | Enterprise SLA |
| **GCP** | T4 (16GB)* | ~$0.35/hr | *Only for INT4-quantized model |

> **Note:** For Qwen2.5-7B in fp16, you need ≥24 GB VRAM. If your budget is tight, use AWQ/GPTQ quantized models which run on 16 GB (T4).

---

## 3. Server Setup

SSH into your VPS:

```bash
ssh root@<your-vps-ip>
```

Update packages and install essentials:

```bash
apt-get update && apt-get upgrade -y
apt-get install -y \
    git curl wget unzip tmux htop \
    build-essential python3-pip python3-venv \
    nvtop                  # GPU monitor (like htop for GPUs)
```

---

## 4. Install CUDA & Python Environment

### 4a. Install CUDA 12.1 (if not pre-installed by provider)

Most GPU cloud providers give you a CUDA-ready image. Verify first:

```bash
nvidia-smi
# Should show CUDA Version: 12.x
```

If CUDA is missing:

```bash
# Download CUDA 12.1 toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # confirm
```

### 4b. Create Python virtual environment

```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

---

## 5. Install vLLM

```bash
# Install vLLM (includes PyTorch with CUDA)
pip install vllm

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

> **Tip:** vLLM ships with its own PyTorch build. Do **not** install a separate `torch` package — this can cause CUDA version conflicts.

Optional — install FlashAttention for extra speed (requires Ampere+ GPU):

```bash
pip install flash-attn --no-build-isolation
```

---

## 6. Upload the Fine-Tuned Model

You have two options:

### Option A — Directly from Hugging Face Hub (Recommended)

If you pushed your merged model to HF Hub:

```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token

# Download the model to the server
huggingface-cli download your-username/vedaz-qwen2.5-7b \
    --local-dir /models/vedaz-astro \
    --local-dir-use-symlinks False
```

### Option B — rsync from your local machine / Colab

```bash
# From your LOCAL machine or Colab, sync the merged model folder:
rsync -avzP --progress \
    ./outputs/vedaz-qwen-merged/ \
    root@<your-vps-ip>:/models/vedaz-astro/
```

### Verify the model directory

```bash
ls /models/vedaz-astro/
# Should contain:
# config.json   tokenizer.json   tokenizer_config.json
# model-00001-of-0000X.safetensors  ...  (safetensors shards)
```

---

## 7. Serve the Model with vLLM

### Basic launch

```bash
source ~/vllm-env/bin/activate

vllm serve /models/vedaz-astro \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name vedaz-astro \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16
```

### Important flags explained

| Flag | Value | Purpose |
|---|---|---|
| `--host` | `0.0.0.0` | Accept connections from all interfaces |
| `--port` | `8000` | Port to listen on |
| `--served-model-name` | `vedaz-astro` | Name clients use in API calls |
| `--max-model-len` | `4096` | Max context window (tokens) |
| `--gpu-memory-utilization` | `0.90` | Reserve 10% VRAM as safety buffer |
| `--dtype` | `bfloat16` | Precision (bf16 for A100, fp16 for others) |
| `--tensor-parallel-size` | `2` | (Optional) Use 2 GPUs for tensor parallelism |
| `--quantization` | `awq` | (Optional) Use if serving AWQ-quantized model |

### For quantized models (runs on 16GB VRAM)

If you have a T4 (16GB), run the original Qwen model in AWQ format:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --served-model-name vedaz-astro \
    --max-model-len 4096
```

---

## 8. Test the API

Once vLLM is running, test it with curl:

### Health check

```bash
curl http://localhost:8000/health
# Returns: {"status":"ok"}
```

### List available models

```bash
curl http://localhost:8000/v1/models
```

### Chat completion (OpenAI format)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vedaz-astro",
    "messages": [
      {
        "role": "system",
        "content": "You are Vedaz'\''s AI Vedic astrologer. You give compassionate, balanced, non-fatalistic guidance."
      },
      {
        "role": "user",
        "content": "Mera breakup ho gaya. Kya kundli mein kuch batao?"
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Python client example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<your-vps-ip>:8000/v1",
    api_key="not-needed",  # vLLM doesn't require auth by default
)

response = client.chat.completions.create(
    model="vedaz-astro",
    messages=[
        {
            "role": "system",
            "content": "You are Vedaz's AI Vedic astrologer. Give compassionate, non-fatalistic guidance."
        },
        {
            "role": "user",
            "content": "Meri shaadi kab hogi? DOB: 4 June 1997, 5:50 AM, Ranchi."
        }
    ],
    max_tokens=512,
    temperature=0.7,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

---

## 9. Production Setup (Systemd + Nginx + SSL)

### 9a. Systemd service (auto-restart on crash/reboot)

Create the service file:

```bash
cat > /etc/systemd/system/vedaz-vllm.service << 'EOF'
[Unit]
Description=Vedaz vLLM Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/models
Environment="PATH=/root/vllm-env/bin:/usr/local/cuda/bin:/usr/bin:/bin"
ExecStart=/root/vllm-env/bin/vllm serve /models/vedaz-astro \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name vedaz-astro \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16
Restart=always
RestartSec=10
StandardOutput=append:/var/log/vedaz-vllm.log
StandardError=append:/var/log/vedaz-vllm.log

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start:

```bash
systemctl daemon-reload
systemctl enable vedaz-vllm
systemctl start vedaz-vllm
systemctl status vedaz-vllm

# Follow logs
journalctl -u vedaz-vllm -f
```

> **Note:** In the systemd unit, `--host 127.0.0.1` binds vLLM to localhost only. Nginx (below) will proxy public traffic to it — this is more secure than exposing vLLM directly.

### 9b. Nginx reverse proxy

Install Nginx:

```bash
apt-get install -y nginx
```

Create a site config:

```bash
cat > /etc/nginx/sites-available/vedaz-api << 'EOF'
server {
    listen 80;
    server_name api.yourdomain.com;    # Replace with your domain or VPS IP

    # Rate limiting (optional but recommended)
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;
    limit_req zone=api burst=10 nodelay;

    location /v1/ {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;        # LLM responses can be slow
        proxy_buffering    off;         # Required for streaming (SSE)
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
EOF

ln -s /etc/nginx/sites-available/vedaz-api /etc/nginx/sites-enabled/
nginx -t           # test config
systemctl restart nginx
```

### 9c. SSL with Let's Encrypt (free HTTPS)

```bash
apt-get install -y certbot python3-certbot-nginx

# Get certificate (replace with your domain)
certbot --nginx -d api.yourdomain.com --non-interactive \
    --agree-tos --email your@email.com

# Auto-renew (certbot adds this to cron by default)
certbot renew --dry-run
```

After this, your API is available at:
```
https://api.yourdomain.com/v1/chat/completions
```

### 9d. API authentication (optional)

vLLM supports a simple API key via `--api-key`:

```bash
# In your systemd ExecStart, add:
--api-key "your-secret-api-key-here"
```

Clients then pass: `Authorization: Bearer your-secret-api-key-here`

---

## 10. Monitoring & Scaling

### Real-time GPU monitoring

```bash
nvtop          # interactive GPU monitor
# OR
watch -n 1 nvidia-smi
```

### vLLM metrics (Prometheus-compatible)

vLLM exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
# Key metrics:
#   vllm:gpu_cache_usage_perc   — KV cache utilization
#   vllm:num_requests_running   — Active requests
#   vllm:request_success_total  — Total successful requests
```

### Scaling options

| Scenario | Solution |
|---|---|
| More throughput on same GPU | Increase `--max-num-seqs` |
| Multiple GPUs (1 machine) | `--tensor-parallel-size 2` |
| Multiple machines | Use `--pipeline-parallel-size` + Ray cluster |
| High availability | Run 2 instances behind Nginx `upstream` load balancer |

---

## 11. Cost Estimates

| Setup | Monthly Cost (24/7) |
|---|---|
| RunPod A10G (24GB) | ~$267/month |
| Lambda Labs A10 | ~$432/month |
| Vast.ai RTX 3090 | ~$150–$290/month |
| AWS g5.xlarge (A10G) | ~$720/month |

> **Cost-saving tip:** Use spot/on-demand instances and auto-shutdown when idle. RunPod and Vast.ai support this. For a demo/assessment, spin up only when needed (~$2–5 total).

---

## 12. Troubleshooting

| Error | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Model too large for VRAM | Reduce `--gpu-memory-utilization` to 0.80, or use `--quantization awq` |
| `ValueError: max_model_len too large` | KV cache doesn't fit | Reduce `--max-model-len` to 2048 |
| `Connection refused` on port 8000 | Firewall blocking | `ufw allow 8000` or use Nginx on port 80/443 |
| Slow first response (~30s) | Model loading | Normal on first request; use `--preemption-mode recompute` |
| `ImportError: flash_attn` | Flash Attention not installed | Pass `--dtype float16` without flash attn, or install it |
| `Tokenizer not found` | Model dir incomplete | Ensure `tokenizer.json` and `config.json` are in model dir |

---

## Summary

```
Local Machine / Colab
      │
      │  1. Fine-tune Qwen2.5-7B with QLoRA (finetune.py)
      │  2. Merge LoRA → full model (outputs/vedaz-qwen-merged/)
      │  3. Upload to VPS via rsync or HF Hub
      ▼
GPU VPS (Ubuntu 22.04 + CUDA 12.1)
      │
      │  4. pip install vllm
      │  5. vllm serve /models/vedaz-astro
      │  6. Systemd service → auto-restart
      │  7. Nginx → reverse proxy + SSL
      ▼
Public API
  https://api.yourdomain.com/v1/chat/completions
  (OpenAI-compatible — works with any OpenAI SDK)
```
