# Multimodal Movie Recommendation System (CV + NLP + GNN)

> Three deep learning pipelines on the MovieLens dataset: genre classification from **posters** (Computer Vision), genre classification from **plot summaries** (NLP), and **user–movie link prediction** (Graph Neural Networks).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-red)
![Task](https://img.shields.io/badge/tasks-CV%20%7C%20NLP%20%7C%20GNN-green)

NTUA — Deep Learning lab project. Built with [Spyridon Evangelatos](https://github.com/<collaborator-handle>) (AM 03400290).

## Results

| Task | Model | Metric | Result |
|------|-------|--------|--------|
| CV — genre from posters | Custom CNN → fine-tuned **ResNet-18** | Macro F1 | **0.26** (+80% vs. CNN) |
| NLP — genre from plots | GRU baseline → **BiGRU + GloVe + Attention** | Macro F1 | **0.41** (+42% vs. baseline) |
| GNN — link prediction | **GraphSAGE** (bipartite user–movie graph) | AUC | **0.93** |

## Key findings

- **ResNet-18 overfits faster than the custom CNN** despite pretraining — best validation checkpoint lands at epoch 1, so early stopping is essential.
- **Graph topology dominates over node features** in dense bipartite recommendation graphs: random-feature baselines nearly match feature-rich variants; only pretrained semantic embeddings give a consistent lift.
- **Multimodal fusion did not beat the best unimodal model**, traced to a projection bottleneck (1024→64) that compresses away signal.

## Tasks in detail

1. **Computer Vision** — multi-label genre classification from movie posters. Custom CNN vs. transfer-learned ResNet-18, with validation-loss tracking and early stopping.
2. **NLP** — multi-label genre classification from plot summaries. Additive-attention GRU baseline vs. an improved Bidirectional GRU with GloVe embeddings.
3. **GNN** — link prediction on a bipartite user–movie graph with GraphSAGE; multi-seed evaluation and late-vs-early fusion ablations.

## Data

Uses the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (25M for CV/NLP training, ml-small for the graph task). Data is **not** committed — download it from the link above and place it under `data/`.

## Quickstart

```bash
git clone https://github.com/<handle>/multimodal-movie-recommender.git
cd multimodal-movie-recommender
pip install -r requirements.txt
# then open the notebook (Colab-ready) and run the cells per task
```

## Repo contents

- `notebook.ipynb` — full pipeline for all three tasks (Colab-compatible)
- per-task model definitions, training loops, and evaluation
- ablation cells: per-class F1, attention visualization, fusion comparison, multi-seed runs
