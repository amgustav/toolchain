# Toolchain

A collection of small, fine-tuned function-calling models built to run cheaply in agent loops.

## Problem
Frontier models like GPT-4 are expensive for high-volume agent loops. Small models are cheaper but struggle with reliable function calling. Toolchain fixes that.

## Models
- [toolchain-qwen2.5-3b](https://huggingface.co/amgustav/forge-qwen2.5-3b-function-calling) — Qwen2.5-3B fine-tuned on function calling

## Demo
Try it: [huggingface.co/spaces/amgustav/forge-qwen2.5-3b-function-calling](https://huggingface.co/spaces/amgustav/forge-qwen2.5-3b-function-calling)

## Training
- Base model: Qwen2.5-3B-Instruct
- Method: QLoRA via Unsloth on free Google Colab T4
- Dataset: NousResearch/hermes-function-calling-v1 + 100 custom examples
- Training time: ~34 minutes

## Reproduce
See `train.py` for full training code.

## Dataset
Custom examples in `data/custom_dataset.json`
