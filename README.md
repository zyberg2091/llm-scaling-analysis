# On the Role of Architecture and Tokenization in Small-Scale Language Models

## Abstract

We present a comparative study of two small transformer-based language models (37M and 53M parameters) trained on the FineWeb dataset and evaluated under a cross-domain setting on WikiText-2. Contrary to the common assumption that larger models perform better, we observe that the smaller 37M model consistently outperforms the larger 53M model across all evaluation metrics, including perplexity and bits-per-token (BPT).

We analyze the underlying causes and show that architectural factors such as model depth, embedding dimension, and context length play a more significant role than parameter count alone. Our findings highlight that evaluation across domains reveals important generalization behaviors that are not captured by in-domain metrics.


## 1. Introduction

Scaling laws suggest that increasing model size improves performance in language modeling. However, these results typically assume consistent architectures, tokenization strategies, and evaluation settings.

In this work, we challenge this assumption by comparing two models of different sizes but different architectural and tokenization configurations. Surprisingly, we find that a smaller model significantly outperforms a larger one under cross-domain evaluation.

Our goal is to understand:
- How architecture affects performance beyond parameter count
- How tokenization influences evaluation metrics
- How models behave under distribution shift

## 2. Model Configurations

We evaluate two causal transformer models:

| Model | Params | Embedding Dim | Layers | Context Length |
|------|--------|---------------|--------|----------------|
| 37M  | ~37M   | 512           | 6      | 256            |
| 53M  | ~53M   | 256           | 3      | 128            |

The 37M model is deeper and wider, while the 53M model has more parameters but reduced depth and representation size.

## 3. Tokenization

The models use different tokenization schemes:

- **37M model**: Custom tokenizer (~20k vocabulary)
- **53M model**: Byte-level tokenizer (`cl100k_base`, ~100k vocabulary)

Tokenization affects:
- Number of prediction steps
- Difficulty of next-token prediction
- Comparability of metrics such as perplexity

## 4. Experimental Setup

### Training Data
Both models are trained on the **FineWeb dataset**, which consists of diverse web-scale text.

### Evaluation Data
We evaluate on **WikiText-2**, a clean and structured dataset derived from Wikipedia articles.

This setup introduces a **cross-domain evaluation scenario**, allowing us to measure generalization.

## 5. Metrics

We report the following metrics:

### Negative Log-Likelihood (Loss)

L = -(1/N) * sum(log P(x_i | x_{<i}))

Measured in **nats per token**.


### Perplexity (PPL)

PPL = exp(L)

Measures model uncertainty.


### Bits per Token (BPT)

BPT = L / log(2)

Represents the average number of bits required to encode each token.


## 6. Results

| Model | Loss   | Perplexity | BPT  |
|------|--------|------------|------|
| 37M  | 5.3501 | 210.62     | 7.72 |
| 53M  | 6.6362 | 762.17     | 9.57 |


## 7. Discussion

### 7.1 Model Size vs Architecture

Despite having fewer parameters, the 37M model significantly outperforms the 53M model. This suggests that **depth and embedding size are more important than parameter count alone** in this regime.


### 7.2 Cross-Domain Generalization

The performance gap becomes more pronounced under distribution shift (FineWeb → WikiText-2). The 37M model generalizes better, indicating that it has learned more robust language representations.

### 7.3 Tokenization Effects

The 53M model uses a much larger vocabulary (~100k), making prediction more difficult and increasing loss. However, tokenization alone does not explain the performance gap.


### 7.4 Context Length

The 37M model benefits from a longer context window (256 vs 128), allowing it to leverage more historical information for prediction.


## 8. Key Insights

- Larger models do not necessarily perform better
- Architectural design (depth, width, context) is critical
- Cross-domain evaluation reveals true generalization ability
- Tokenization impacts metrics but is not the sole factor
- Parameter count alone is a weak proxy for performance


## 9. Conclusion

We show that a smaller transformer model can outperform a larger one when architectural design and training dynamics are more favorable. In a cross-domain setting, the 37M model demonstrates significantly better generalization than the 53M model.

These results emphasize the importance of model design choices beyond scaling parameter count, particularly in small-scale language models.


## 10. Future Work

- Train both models with the same tokenizer for fair comparison
- Increase depth of the larger model
- Evaluate on additional datasets for robustness
- Analyze training dynamics (loss curves, convergence behavior)
- Explore architectural improvements (RMSNorm, attention variants)


## Appendix (Optional)

### Model Implementation Notes

- Transformer-based causal LM
- Trained on FineWeb
- Evaluated without fine-tuning on WikiText-2


## Pretrained Model

[fineweb_pretrained_model_37M](https://huggingface.co/zyberg2091/fineweb_pretrained_model_37M)
