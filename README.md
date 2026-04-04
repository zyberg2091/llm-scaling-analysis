# On the Role of Architecture and Tokenization in Small-Scale Language Models

## Abstract

We present an empirical comparison of two small transformer-based language models (37M and 53M parameters) trained on the FineWeb dataset and evaluated under a cross-domain setting on WikiText-2.

While scaling laws suggest that larger models should perform better under controlled settings, we observe that the 37M model achieves lower loss, perplexity, and bits-per-token (BPT) than the 53M model in this setup. However, the models differ not only in parameter count, but also in architectural design (depth, embedding dimension, and context length) and tokenization.

These results indicate that performance differences in small-scale regimes cannot be attributed to parameter count alone, and highlight the influence of architectural and tokenization choices, particularly under distribution shift.

---

## 1. Introduction

Scaling laws in language modeling indicate that increasing parameter count improves performance when other factors are held constant. In practice, however, models often differ along multiple axes, including architecture and tokenization.

In this work, we study a setting where these factors vary simultaneously. We compare two models trained on the same dataset but with different architectural configurations and tokenization schemes, and evaluate them on a different domain.

Our goal is not to isolate a single causal factor, but to examine how these design choices jointly affect performance in a realistic, non-controlled setting.

We focus on:

- The role of architecture beyond parameter count  
- The impact of tokenization on evaluation metrics  
- Generalization under distribution shift  

---

## 2. Model Configurations

We evaluate two causal transformer language models:

| Model | Parameters | Embedding Dim | Layers | Context Length |
|------|------------|---------------|--------|----------------|
| 37M  | ~37M       | 512           | 6      | 256            |
| 53M  | ~53M       | 256           | 3      | 128            |

The 37M model is deeper and wider, while the 53M model allocates parameters differently with reduced depth and embedding dimension.

---

## 3. Tokenization

The models use different tokenization schemes:

- **37M model**: Custom tokenizer (~20k vocabulary)  
- **53M model**: Byte-level tokenizer (`cl100k_base`, ~100k vocabulary)  

Tokenization affects:

- Sequence length (number of prediction steps)  
- Difficulty of next-token prediction  
- Comparability of token-level metrics such as perplexity  

Due to these differences, token-level metrics should be interpreted with caution when comparing across models.

---

## 4. Experimental Setup

### Training Data

Both models are trained on the **FineWeb dataset**, a large-scale web corpus.

### Evaluation Data

We evaluate both models on WikiText-2 without fine-tuning, creating a **cross-domain evaluation setting**.

Both models are evaluated on the same underlying text corresponding to approximately 200k tokens prior to tokenization. Due to differences in tokenization:

- 37M model: 190,965 tokens  
- 53M model: 136,406 tokens  

This discrepancy arises from differences in vocabulary size and token granularity.

---

## 5. Metrics

Let:

- x = (x₁, x₂, ..., x_N): token sequence  
- P(x_i | x_<i): model probability  
- N: number of tokens  

### Negative Log-Likelihood (Loss)

L = -(1/N) * Σ log P(x_i | x_<i)

Measured in **nats per token**.

---

### Perplexity (PPL)

PPL = exp(L)

Measures the effective uncertainty of the model.

---

### Bits per Token (BPT)

BPT = L / log(2)

Represents the average number of bits required to encode each token.

---

## 6. Results

| Model | Loss   | Perplexity | BPT  |
|------|--------|------------|------|
| 37M  | 5.3537 | 211.38     | 7.72 |
| 53M  | 6.6659 | 785.21     | 9.62 |

---

## 7. Discussion

### 7.1 Model Size vs Architecture

Although the 53M model has more parameters, it underperforms relative to the 37M model in this setup. However, this comparison is confounded by differences in depth, embedding dimension, and context length. As a result, the effect of parameter count cannot be isolated from architectural differences.

---

### 7.2 Cross-Domain Generalization

The evaluation is performed under a distribution shift (FineWeb → WikiText-2). The 37M model achieves lower loss in this setting, suggesting that its configuration generalizes more effectively under these conditions.

---

### 7.3 Tokenization Effects

The 53M model uses a significantly larger vocabulary (~100k vs ~20k), which increases the difficulty of next-token prediction and affects token-level metrics. While this likely contributes to the observed performance gap, it does not fully account for it.

---

### 7.4 Context Length

The 37M model uses a longer context window (256 vs 128), allowing it to condition on more prior tokens. This may contribute to improved performance, particularly in cross-domain evaluation.

---

## 8. Limitations

- Models differ in multiple factors (architecture, tokenizer, context), preventing causal attribution  
- Token-level metrics are not directly comparable across different tokenizers  
- Evaluation is performed on a single dataset  
- Training hyperparameters are not extensively tuned  

---

## 9. Key Observations

- Parameter count alone is not sufficient to explain performance differences in this setting  
- Architectural configuration (depth, width, context length) appears to play a significant role  
- Tokenization impacts both sequence structure and evaluation metrics  
- Cross-domain evaluation highlights differences in generalization across configurations  

---

## 10. Conclusion

We present a comparative analysis of two small language models under a cross-domain evaluation setting. While the smaller model achieves better performance, the results emphasize the role of architectural and tokenization choices in shaping model behavior.

Future work should focus on controlled experiments to isolate individual factors and better understand their contributions.

---

## 11. Future Work

- Train both models with the same tokenizer for controlled comparison  
- Isolate architectural variables while scaling parameter count  
- Evaluate on additional datasets  
- Analyze training dynamics and convergence behavior  
- Explore architectural variants (e.g., normalization schemes, attention modifications)  

---

## Appendix

### Model Implementation Notes

- Transformer-based causal language model  
- Trained on FineWeb  
- Evaluated on WikiText-2 without fine-tuning  

---

## Pretrained Model

- 37M Model: https://huggingface.co/zyberg2091/fineweb_pretrained_model_37M
