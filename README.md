# On the Role of Architecture and Tokenization in Small-Scale Language Models

## Abstract

We present an empirical comparison of two small transformer-based language models (37M and 53M parameters) trained on the FineWeb dataset and evaluated under a cross-domain setting on WikiText-2.

While scaling laws suggest that larger models should perform better, we observe that the 37M model achieves lower loss, perplexity, and bits-per-token (BPT) than the 53M model. However, the two models differ not only in parameter count, but also in architectural design (depth, embedding dimension, and context length) and tokenization.

We analyze these factors and show that the observed performance differences cannot be attributed to parameter count alone. These results highlight the importance of architectural and tokenization choices in small-scale regimes, particularly under distribution shift.


## 1. Introduction

Scaling laws in language modeling indicate that increasing parameter count improves performance when other factors are held constant. However, in practical settings, models often differ along multiple axes, including architecture and tokenization.

In this work, we study a setting where these factors vary simultaneously. We compare two models trained on the same dataset but with different architectural configurations and tokenization schemes, and evaluate them on a different domain.

Our goal is not to isolate a single causal factor, but to understand how these design choices jointly affect performance.

We focus on:
- The role of architecture beyond parameter count  
- The impact of tokenization on evaluation metrics  
- Generalization under distribution shift  


## 2. Model Configurations

We evaluate two causal transformer language models:

| Model | Parameters | Embedding Dim | Layers | Context Length |
|------|------------|---------------|--------|----------------|
| 37M  | ~37M       | 512           | 6      | 256            |
| 53M  | ~53M       | 256           | 3      | 128            |

The 37M model is deeper and wider, while the 53M model allocates parameters differently with reduced depth and embedding dimension.


## 3. Tokenization

The models use different tokenization schemes:

- **37M model**: Custom tokenizer (~20k vocabulary)  
- **53M model**: Byte-level tokenizer (`cl100k_base`, ~100k vocabulary)  

Tokenization affects:
- Sequence length (number of prediction steps)  
- Difficulty of next-token prediction  
- Comparability of token-level metrics such as perplexity  

Due to these differences, direct comparison of token-level metrics must be interpreted with caution.


## 4. Experimental Setup

### Training Data

Both models are trained on the **FineWeb dataset**, a large-scale web corpus.

### Evaluation Data

We evaluate both models on WikiText-2 without fine-tuning, creating a **cross-domain evaluation setting**.

Both models are evaluated on the same underlying text corresponding to approximately 200k tokens prior to tokenization. Due to differences in tokenization:

- 37M model: 190,965 tokens  
- 53M model: 136,406 tokens  

This discrepancy arises from differences in vocabulary size and token granularity.


## 5. Metrics

Let:

- x = (x₁, x₂, ..., x_N): token sequence  
- P(x_i | x_<i): model probability  
- N: number of tokens  

### Negative Log-Likelihood (Loss)

L = -(1/N) * Σ_{i=1}^{N} log P(x_i | x_<i)

Measured in **nats per token**.


### Perplexity (PPL)

PPL = exp(L)

Measures the effective uncertainty of the model.


### Bits per Token (BPT)

BPT = L / log(2)

Represents the average number of bits required to encode each token.


## 6. Results

| Model | Loss   | Perplexity | BPT  |
|------|--------|------------|------|
| 37M  | 5.3537 | 211.38     | 7.72 |
| 53M  | 6.6659 | 785.21     | 9.62 |


## 7. Discussion

### 7.1 Model Size vs Architecture

Although the 53M model has more parameters, it underperforms relative to the 37M model. However, this comparison is confounded by differences in depth, embedding dimension, and context length. Therefore, the results do not isolate the effect of parameter count alone.


### 7.2 Cross-Domain Generalization

The evaluation is performed under a distribution shift (FineWeb → WikiText-2). The 37M model achieves lower loss in this setting, suggesting stronger generalization under the given configuration.


### 7.3 Tokenization Effects

The 53M model uses a significantly larger vocabulary (~100k vs ~20k), increasing the difficulty of next-token prediction. This contributes to higher loss, though it does not fully explain the observed performance gap.


### 7.4 Context Length

The 37M model uses a longer context window (256 vs 128), allowing it to leverage more historical information during prediction, which may contribute to improved performance.


## 8. Limitations

- Models differ in multiple factors (architecture, tokenizer, context), making causal attribution difficult  
- Token-level metrics are not directly comparable across different tokenizers  
- Evaluation is performed on a single dataset  
- Training hyperparameters are not extensively tuned  


## 9. Key Insights

- Parameter count alone is not a sufficient predictor of performance  
- Architectural design (depth, width, context length) plays a significant role  
- Tokenization strongly influences evaluation metrics  
- Cross-domain evaluation reveals differences in generalization  


## 10. Conclusion

We present a comparative analysis of two small language models under a cross-domain evaluation setting. While the smaller model achieves better performance, the results highlight the importance of architectural and tokenization choices rather than parameter count alone.

Future work should focus on controlled experiments where individual factors are isolated.


## 11. Future Work

- Train both models with the same tokenizer for fair comparison  
- Control architectural variables while scaling parameter count  
- Evaluate on additional datasets  
- Analyze training dynamics and convergence behavior  
- Explore architectural improvements (e.g., RMSNorm, attention variants)  


## Appendix

### Model Implementation Notes

- Transformer-based causal language model  
- Trained on FineWeb  
- Evaluated on WikiText-2 without fine-tuning  


## Pretrained Model

- 37M Model: https://huggingface.co/zyberg2091/fineweb_pretrained_model_37M
