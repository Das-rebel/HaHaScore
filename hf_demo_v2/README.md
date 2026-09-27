---
title: HaHaScore Cascade v8
emoji: 🎭
colorFrom: purple
colorTo: red
sdk: static
pinned: false
license: mit
---

# HaHaScore Cascade v8 — YouTube Humor Detector

AI-powered humor detection for YouTube videos. Paste a YouTube URL and get instant funniness analysis per segment.

## ✨ Features

- 🎬 **YouTube URL support** — paste any video link
- 📊 **20-segment analysis** — per-segment funniness scores
- 🔥 **Peak detection** — find the funniest moments
- 📈 **Visual timeline** — color-coded humor heatmap
- 💬 **Color-coded transcript** — see which phrases scored high
- ⚡ **Real-time inference** — <3ms per video

## 🧠 Architecture

- **Pretrain**: DistilBERT on 30K Reddit jokes
- **Fine-tune**: Cascade Gate on 639 standup comedy files
- **Inference**: ONNX INT8 (2.9 MB, <3ms CPU)
- **Val AUC**: 0.802

## 💡 How to Use

1. Paste any YouTube URL in the input box
2. Click "Analyze"
3. View per-segment scores, peaks, and insights

## 🔗 Links

- [GitHub Repo](https://github.com/Das-rebel/HaHaScore)
- [HuggingFace Model](https://huggingface.co/Hayasuki/hahascore-cascade)
