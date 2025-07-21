# Unicode 1

1. `'\x00'`
2. `print(chr(0))` outputs nothing but an empty line.
3. It simply disappears when printed but keeps the original '\x00' code if not.

# Unicode 2

1. UTF-8 will encode the string in the most shortest length, which may help save the cost.
2. One byte doesn't correponds to one Unicode character necessarily. Any unicode character encoded by multiple bytes in UTF-8 is a counter-example, such as the Chinese character '坤'.
3. `b'\xe4\xbd'`

# transformer_accounting

1.  1. RMSNorm: $d_{model}=1600$
    2. SwiGLU: $3\times d_{model}\times d_ff=30,720,000$
    3. MHA: $4\times d_{model}^2=10,240,000$
    4. Transformer Block: $2 RMSNorm+1 MHA+1 FFN=40,963,200$
    5. Linear: $d_{model}\times vocabSize=804,112,000$

Total learnable params: $num_{layers}\times Block+RMSNorm+Linear=2,770,347,200$

Memory:

$$
\frac{params\times 4}{1024*1024*1024}=13.3 GB
$$

2.  Matrix multiplies for a forward pass with context length T = 1024:
    - **FLOPs per Transformer Block:**
      - SwiGLU: $T \times (6 \times d_{model} \times d_{ff}) = 1024 \times (6 \times 1600 \times 6400) \approx 6.29 \times 10^{10}$
      - MHA: $8T d_{model}^2 + 4T^2 d_{model} = (8 \times 1024 \times 1600^2) + (4 \times 1024^2 \times 1600) \approx 2.10 \times 10^{10} + 6.71 \times 10^9 \approx 2.77 \times 10^{10}$
      - RMSNorm: $2 \times (T \times d_{model}) = 2 \times (1024 \times 1600) \approx 3.28 \times 10^6$
      - Total per Block: $SwiGLU + MHA + 2 \times Norm \approx 6.29 \times 10^{10} + 2.77 \times 10^{10} + 3.28 \times 10^6 \approx 9.06 \times 10^{10}$
    - **FLOPs for entire model:**
      - All Transformer Blocks: $num_{layers} \times \text{FLOPs per Block} = 64 \times 9.06 \times 10^{10} \approx 5.80 \times 10^{12}$
      - Final Output Layer: $T \times (2 \times d_{model} \times vocabSize) = 1024 \times (2 \times 1600 \times 50257) \approx 1.65 \times 10^{11}$
      - Final RMSNorm: $T \times d_{model} \approx 1.64 \times 10^6$
    - **Total FLOPs:** $\approx 5.80 \times 10^{12} + 1.65 \times 10^{11} \approx 5.97 \times 10^{12}$ (approx. 6 Trillion FLOPs)
3.  Transformer Blocks
4.  GPT-2 small (12 layers, 768 d_model, 12 heads, 3072 d_ff)

- Parameters:

  - Token + Positional Embeddings: $(50257 + 1024) \times 768 = 39,403,008$
  - Transformer Block: $12 \times 768^2 = 7,077,888$
  - Total: $39,403,008 + 12 \times (7,077,888 + 4 \times 768) + 2 \times 768 = 124,439,808$ (approx. 124M)

- FLOPs (forward pass with context length T = 1024):
  - MHA per Block: $8T d_{model}^2 + 4T^2 d_{model} = (8 \times 1024 \times 768^2) + (4 \times 1024^2 \times 768) \approx 4.83 \times 10^9 + 3.22 \times 10^9 \approx 8.05 \times 10^9$
  - FFN per Block: $16T d_{model}^2 = 16 \times 1024 \times 768^2 \approx 9.66 \times 10^9$
  - All Blocks: $12 \times (8.05 \times 10^9 + 9.66 \times 10^9) \approx 2.13 \times 10^{11}$
  - Output Layer: $2 \times T \times d_{model} \times vocabSize = 2 \times 1024 \times 768 \times 50257 \approx 7.89 \times 10^{10}$
  - Total FLOPs: $\approx 2.13 \times 10^{11} + 7.89 \times 10^{10} \approx 2.92 \times 10^{11}$

GPT-2 medium (24 layers, 1024 d_model, 16 heads, 4096 d_ff)

- Parameters:

  - Token + Positional Embeddings: $(50257 + 1024) \times 1024 = 52,511,744$
  - Transformer Block: $12 \times 1024^2 = 12,582,912$
  - Total: $52,511,744 + 24 \times (12,582,912 + 4 \times 1024) + 2 \times 1024 = 354,824,192$ (approx. 355M)

- FLOPs (forward pass with context length T = 1024):
  - MHA per Block: $8T d_{model}^2 + 4T^2 d_{model} = (8 \times 1024 \times 1024^2) + (4 \times 1024^2 \times 1024) \approx 8.59 \times 10^9 + 4.30 \times 10^9 \approx 1.29 \times 10^{10}$
  - FFN per Block: $16T d_{model}^2 = 16 \times 1024 \times 1024^2 \approx 1.72 \times 10^{10}$
  - All Blocks: $24 \times (1.29 \times 10^{10} + 1.72 \times 10^{10}) \approx 7.22 \times 10^{11}$
  - Output Layer: $2 \times T \times d_{model} \times vocabSize = 2 \times 1024 \times 1024 \times 50257 \approx 1.05 \times 10^{11}$
  - Total FLOPs: $\approx 7.22 \times 10^{11} + 1.05 \times 10^{11} \approx 8.27 \times 10^{11}$

GPT-2 large (36 layers, 1280 d_model, 20 heads, 5120 d_ff)

- Parameters:

  - Token + Positional Embeddings: $(50257 + 1024) \times 1280 = 65,639,680$
  - Transformer Block: $12 \times 1280^2 = 19,660,800$
  - Total: $65,639,680 + 36 \times (19,660,800 + 4 \times 1280) + 2 \times 1280 = 774,036,480$ (approx. 774M)

- FLOPs (forward pass with context length T = 1024):
  - MHA per Block: $8T d_{model}^2 + 4T^2 d_{model} = (8 \times 1024 \times 1280^2) + (4 \times 1024^2 \times 1280) \approx 1.34 \times 10^{10} + 5.37 \times 10^9 \approx 1.88 \times 10^{10}$
  - FFN per Block: $16T d_{model}^2 = 16 \times 1024 \times 1280^2 \approx 2.68 \times 10^{10}$
  - All Blocks: $36 \times (1.88 \times 10^{10} + 2.68 \times 10^{10}) \approx 1.64 \times 10^{12}$
  - Output Layer: $2 \times T \times d_{model} \times vocabSize = 2 \times 1024 \times 1280 \times 50257 \approx 1.32 \times 10^{11}$
  - Total FLOPs: $\approx 1.64 \times 10^{12} + 1.32 \times 10^{11} \approx 1.77 \times 10^{12}$

As model size increases, the Transformer blocks (MHA and FFN) take up a proportionally larger share of the total FLOPs compared to the embedding and output layers. The FLOPs in the blocks scale with $O(d_{model}^2)$, while other parts scale with $O(d_{model})$, making the blocks dominant for larger models.

5. Yes, the total FLOPs and the relative contributions change dramatically when the context length increases. The core reason is that the self-attention mechanism has a computational complexity of $O(T^2 \cdot d_{model})$ with respect to the context length $T$, while the Feed-Forward Network (FFN) layers have a linear complexity of $O(T \cdot d_{model}^2)$.

Let's analyze this for GPT-2 XL ($d_{model}=1600$, 48 layers).

- **FLOPs per Transformer Block:**

  - **Attention:** $F_{attn}(T) \approx 4T^2d_{model} + 8Td_{model}^2$
  - **FFN:** $F_{ffn}(T) \approx 16Td_{model}^2$

- **Case 1: Standard Context (T = 1024)**

  - The FFN part is dominant.
  - $F_{attn}(1024) \approx 4 \cdot 1024^2 \cdot 1600 + 8 \cdot 1024 \cdot 1600^2 \approx 6.7 \times 10^9 + 2.1 \times 10^{10} \approx 2.77 \times 10^{10}$
  - $F_{ffn}(1024) \approx 16 \cdot 1024 \cdot 1600^2 \approx 4.2 \times 10^{10}$
  - **Relative contribution (within a block):** Attention is ~40%, FFN is ~60%.

- **Case 2: Long Context (T = 16,384)**
  - The $T^2$ term in attention makes it dominant.
  - $F_{attn}(16384) \approx 4 \cdot 16384^2 \cdot 1600 + 8 \cdot 16384 \cdot 1600^2 \approx 1.72 \times 10^{12} + 3.35 \times 10^{11} \approx 2.05 \times 10^{12}$
  - $F_{ffn}(16384) \approx 16 \cdot 16384 \cdot 1600^2 \approx 6.7 \times 10^{11}$
  - **Relative contribution (within a block):** Attention becomes ~75%, FFN becomes ~25%.

**Conclusion:**
When increasing the context length from 1024 to 16384 (a 16x increase), the total FLOPs for a forward pass increase by approximately **38x**. The self-attention computation, which was a smaller part of the block's computation, becomes the dominant component, consuming the vast majority of FLOPs.

# Learning_rate_tuning
1. When lr is 1e1 or 1e2, the loss keeps to be 0.
2. When lr is le3, it explodes to infinity.

# adamwAccounting
1. Decompose by parts:
> vocab_size:$v$, d_model:$d$, $num_heads: h$, $num_layers: n$, $context_length: s$, batch_size: $b$
   - parameters: 
     - Embedding: $vd$
     - Norm: $d$
     - MHA: $4d^2$
     - FFN: $3d^(4d)=12d^2$
     - Block: 2Norm+MHA+FFN=$2d+16d^2$
     - Linear: $vd$
     - Total: $vd+n(2d+16d^2)+d+vd=2vd+(2n+1)d+16nd^2$
   - activations:
     - RMSNorm: $bsd$
     - MHA
       - QKV: $3bsd$
       - QK: $bhs^2$
       - softmax: $bhs^2$
       - weighted sum: $bsd$
       - output: $bsd$
       - total: $5bsd+2bhs^2$
     - FFN: $4bsd+4bsd+bsd=9bsd$
     - block: 2Norm+FFN+MHA=2bsd+9bsd+(5bsd+2bs^2)=16bsd+2bhs^2
     - output embedding: $bsv$
     - cross-entropy: $bsv$
     - total: $n(16bsd+2bhs^2)+bsd+2bsv=(16n+1)bsd+2nbhs^2+2bsv$
   - gradients: the same as the parameters
   - optinmizer state: nearly twice as the parameters
   - total: $4(2vd+(2n+1)d+16nd^2)+(16n+1)bsd+2nbhs^2+2bsv$
Then multiply by a coefficent of $4/(2^{30})=2^{-28}GB$
2. $5.2b+31.7(GB)$
So the maximum batch size within a 80GB memory is around 10
3. twice as the forward FLOPs
4. to be done