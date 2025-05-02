Implementation Plan: “Fine-Tuning GPT-2 for Game Decision-Making Using HuggingFace”

────────────────────────────────────────
0.  High-Level Objectives
• Build a reproducible pipeline that ingests turn-based game data, fine-tunes `gpt2-medium`, and outputs decisions conditioned on game state.  
• Fit on a single A100 / 3090-class GPU by using parameter-efficient methods (LoRA, layer freezing, gradient-checkpointing).  
• Deliver: (1) cleaned datasets, (2) fine-tuned checkpoints, (3) evaluation harness + metrics, (4) demo notebook / REST API, (5) documentation.

────────────────────────────────────────
1.  Project Setup (Week 1)

1.1 Repository & CI
  – Create Git repo with Poetry or Conda environment (`python-3.10`).  
  – Pre-commit hooks: black, isort, ruff, nbstripout.  
  – GitHub Actions: lint, unit tests, CUDA test-run (tiny batch) on push.

1.2 Dependencies  
```bash
pip install transformers datasets peft bitsandbytes deepspeed accelerate
pip install wandb evaluate sacrebleu rouge_score pandas pydantic
```

1.3 Hardware Assumptions  
  – Dev: single 24 GB GPU.  
  – Training: 1×A100 80 GB or 2×A100 40 GB (FP16) with DeepSpeed-ZeRO-3.

────────────────────────────────────────
2.  Data Pipeline (Weeks 2–4)

2.1 Source Datasets  
  a) Turn-Based Decision Logs (chess/Go/strategy games).  
  b) Text-based game narratives & interactive fiction.  
  c) (Optional) “狼人杀” or similar social-deduction logs for additional diversity.  

2.2 Collection Tasks  
  – Scrape PGN/SGF files or public APIs (e.g., lichess, KGS).  
  – Parse to JSON: `{state_repr, move, meta}`.  
  – Mine dialogues from interactive fiction repos (Adventure-stories, ChoiceScript).  

2.3 Schema (Pydantic)  
```
GameExample:
    game_id: str
    turn_idx: int
    state: str        # textual or compressed board FEN/SGF snippet
    legal_moves: str  # optional; may aid evaluation
    decision: str     # ground-truth move or action
```

2.4 Cleaning & Splits  
  – Remove duplicates, illegal moves ⇒ heuristics + engines.  
  – Stratified split 80/10/10 train/val/test by game_id.  
  – Save to HuggingFace `datasets` arrow files.

2.5 Data Augmentation  
  – Random perspective flips (board games), synonym substitution (narratives).  
  – “Dynamic mask”: randomly hide non-essential parts of state description.

Deliverables: `data/README.md`, dataset loading script (`datasets.load_dataset("local")`).

────────────────────────────────────────
3.  Baseline & Training Infrastructure (Weeks 3–5 overlap)

3.1 Model Wrapper  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
```

3.2 Parameter-Efficient Training  
```python
from peft import LoraConfig, get_peft_model
peft_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["c_attn"], lora_dropout=0.1)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
```
  – Freeze bottom N=12 transformer blocks (empirically chosen after pilot).  
  – Enable gradient checkpointing (`model.gradient_checkpointing_enable()`).  
  – Mixed precision (`fp16` or `bfloat16`).

3.3 DeepSpeed Config (`ds_config.json`) with ZeRO-3 offload for >40 GB cases.

────────────────────────────────────────
4.  Experiment Schedule (Weeks 5–9)

| Exp | Layers Frozen | LoRA r/α | LR  | Epochs | Notes                 |
|-----|--------------|----------|-----|--------|-----------------------|
| B0  | 0            | –        | 3e-5| 3      | Full fine-tune (baseline) |
| P1  | 12           | 8/32     | 3e-5| 3      | Preferred (resource)  |
| P2  | 12           | 16/32    | 1e-5| 5      | Lower LR, more epochs |
| P3  | 6            | 8/16     | 3e-5| 3      | Fewer frozen layers   |

Track experiments with Weights & Biases (`wandb`), tagging git commit + dataset hash.

────────────────────────────────────────
5.  Evaluation Framework (Weeks 5–9 parallel)

5.1 Automatic Metrics  
  – Accuracy / Top-k accuracy on exact move prediction.  
  – BLEU / ROUGE-L for narrative decision justification strings.  
  – Per-turn perplexity.

5.2 Functionality Tests  
  – Rule compliance: run game engine validator on generated move.  
  – Latency: tokens/sec generation on GPU and CPU.  

5.3 Human/Expert Study (stretch)  
  – Recruit 5–10 players to rate decision quality (Likert 1–5).  

Implement `evaluate.py` runnable CLI:  
```bash
python evaluate.py --model_path ckpts/exp_P1 --split test
```

────────────────────────────────────────
6.  Iterative Optimisation (Weeks 9–12)

6.1 Error Analysis  
  – Mis-predicted states grouped by feature (opening, midgame, endgame).  
  – Confusion heatmaps for illegal vs. sub-optimal actions.  

6.2 Curriculum / Progressive Training  
  – Start with short state descriptions; gradually increase move depth.  

6.3 Knowledge Distillation (optional)  
  – Distil P1 into `gpt2` (124 M) for lightweight deployment.

────────────────────────────────────────
7.  Packaging & Deployment (Weeks 12–14)

7.1 Model Card + Weights  
  – Push to HF Hub with license, intended-use, data citation.

7.2 Inference API  
```python
from fastapi import FastAPI
app = FastAPI()
@app.post("/predict")   # receives JSON state, returns decision
def predict(payload): ...
```
Dockerfile with NVIDIA base image, gunicorn, uvicorn workers.

7.3 Demo Notebook  
  – Colab notebook that loads model, visualises board, calls API.

────────────────────────────────────────
8.  Documentation & Deliverables (Weeks 14–15)

• `docs/` MkDocs site: setup, reproduce, extend.  
• Final report: experiment table, charts, ablation results, qualitative examples.  
• Slide deck for project review.

────────────────────────────────────────
9.  Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Dataset noise / illegal moves | Use rule engines for validation; filter. |
| GPU memory limits | LoRA + ZeRO-3 + checkpointing; gradient accumulation. |
| Over-fitting small datasets | Data augmentation, early stopping, eval on unseen games. |
| Evaluation subjectivity | Combine objective engine-based scores with human study. |

────────────────────────────────────────
10.  Timeline Overview

Week 1   Setup & CI  
Weeks 2-4   Data collection & cleaning  
Weeks 3-5   Baseline infra (overlap)  
Weeks 5-9   Training experiments + evaluation harness  
Weeks 9-12  Optimisation + error analysis  
Weeks 12-14  Packaging, API, demo  
Week 15   Final report & presentation  

Total: ~15 weeks (≈ 1 academic semester).

This plan turns the proposal into a concrete, resource-aware roadmap that culminates in a deployable GPT-2 model capable of making strategic decisions in turn-based games.
