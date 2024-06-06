---
language:
- en
license: apache-2.0
base_model: openai/whisper-small
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: whisp-small-hw
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# whisp-small-hw

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0621
- Wer: 6.3204

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss | Wer    |
|:-------------:|:-------:|:----:|:---------------:|:------:|
| 0.1411        | 5.7143  | 1000 | 0.1502          | 7.8871 |
| 0.0566        | 11.4286 | 2000 | 0.0749          | 6.7767 |
| 0.0438        | 17.1429 | 3000 | 0.0650          | 6.3508 |
| 0.0363        | 22.8571 | 4000 | 0.0626          | 6.2747 |
| 0.0324        | 28.5714 | 5000 | 0.0621          | 6.3204 |


### Framework versions

- Transformers 4.41.0
- Pytorch 2.3.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
