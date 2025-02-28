# Instruction Tuning and DPO Mistral-7B

## Overview

This document outlines the training process for fine-tuning **Mistral-7B**, which consists of two stages: **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**. The SFT phase adapts the model to instruction-based responses, while the DPO phase enhances preference alignment. 

## Supervised Fine-Tuning (SFT)

### Dataset

The fine-tuning uses **Databricks Dolly 15K**, a dataset containing **15,011** samples structured with `instruction`, `context`, `response`, and `category` fields. The data is filtered (length < 2500) and split into **90%-10%** for training and testing.

### Training Setup

- **Model:** Mistral-7B with QLoRA
- **Method:** QLoRA (r=16,lora_alpha=32, lora_dropout=0.05, 4-bit quantization)
- **Batch Size:** 1 per device
- **Gradient Accumulation:** 4 steps
- **Learning Rate:** 2e-4
- **Max Steps:** 5000
- **Sequence Length:** 1024 tokens
- **Optimizer:** `paged_adamw_8bit`
- **Precision:** FP16

### Training Log
![SFT traing logging](train_logging\SFT_logging.png))

### Results

The SFT phase concluded with **5000 steps**, achieving a final loss of **1.167**, indicating improved instruction-following capabilities.

## Direct Preference Optimization (DPO)

(To be added)

## References

- Databricks Dolly 15K: [Hugging Face Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- Mistral-7B: [Hugging Face Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- QLoRA: Efficient fine-tuning technique using quantized adapters.

