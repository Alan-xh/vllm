---
title: vLLM 分页注意力
---
[](){ #design-paged-attention }

目前，vLLM 使用其自己的多头查询注意力内核实现（`csrc/attention/attention_kernels.cu`）。该内核设计为与 vLLM 的分页键值（KV）缓存兼容，其中键和值缓存存储在不同的块中（请注意，这里的块概念与 GPU 线程块不同。因此，在后续文档中，我将 vLLM 分页注意力块称为“块”，而将 GPU 线程块称为“线程块”）。

为了实现高性能，该内核依赖于专门设计的内存布局和访问方法，特别是在线程从全局内存读取数据到共享内存时。本文档的目的是逐步提供内核实现的高层次解释，帮助那些希望了解 vLLM 多头查询注意力内核的人。阅读本文档后，用户可能会更好地理解并更容易跟踪实际实现。

请注意，本文档可能不会涵盖所有细节，例如如何计算对应数据的正确索引或点乘的实现。然而，在阅读本文档并熟悉高层次逻辑流程后，阅读实际代码并理解细节应该会更容易。

## 输入

内核函数接受一系列参数，供当前线程执行其分配的工作。三个最重要的参数是输入指针 `q`、`k_cache` 和 `v_cache`，它们分别指向全局内存中需要读取和处理的查询、键和值数据。输出指针 `out` 指向应写入结果的全局内存。这四个指针实际上指向多维数组，但每个线程仅访问分配给它的部分数据。为简单起见，我在此省略了所有其他运行时参数。

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ... // 其他辅助参数
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // 其他辅助参数
)
```

函数签名上方还有一系列在编译时确定的模板参数。`scalar_t` 表示查询、键和值数据元素的数据类型，例如 FP16。`HEAD_SIZE` 表示每个头的元素数量。`BLOCK_SIZE` 表示每个块中的令牌数量。`NUM_THREADS` 表示每个线程块中的线程数量。`PARTITION_SIZE` 表示张量并行 GPU 的数量（为简单起见，我们假设此值为 0，张量并行被禁用）。

有了这些参数，我们需要进行一系列准备工作。这包括计算当前头索引、块索引和其他必要变量。然而，暂时我们可以忽略这些准备工作，直接进入实际计算。一旦我们掌握了整个流程，理解这些准备工作会更容易。

## 概念

在深入计算流程之前，我想先介绍一些后续部分需要用到的概念。如果遇到任何令人困惑的术语，你可以跳过本节，稍后再返回查看。

- **序列**：序列表示客户端请求。例如，`q` 指向的数据形状为 `[num_seqs, num_heads, head_size]`，表示 `q` 指向的总共有 `num_seqs` 个查询序列数据。由于此内核是单查询注意力内核，每个序列只有一个查询令牌。因此，`num_seqs` 等于批处理中处理的令牌总数。
- **上下文**：上下文由序列生成的令牌组成。例如，`["What", "is", "your"]` 是上下文令牌，而输入查询令牌是 `"name"`。模型可能生成令牌 `"?"`。
- **向量（Vec）**：向量是一组同时获取和计算的元素。对于查询和键数据，向量大小（`VEC_SIZE`）确定为每个线程组一次可以获取和计算 16 字节的数据。对于值数据，向量大小（`V_VEC_SIZE`）确定为每个线程一次可以获取和计算 16 字节的数据。例如，如果 `scalar_t` 是 FP16（2 字节）且 `THREAD_GROUP_SIZE` 是 2，则 `VEC_SIZE` 将为 4，而 `V_VEC_SIZE` 将为 8。
- **线程组**：线程组是一小组线程（`THREAD_GROUP_SIZE`），一次处理一个查询令牌和一个键令牌。每个线程仅处理令牌数据的一部分。由一个线程组处理的元素总数称为 `x`。例如，如果线程组包含 2 个线程且头大小为 8，则线程 0 处理索引 0、2、4、6 的查询和键元素，而线程 1 处理索引 1、3、5、7 的元素。
- **块**：vLLM 中的键和值缓存数据被分成块。每个块存储一个头中固定数量（`BLOCK_SIZE`）的令牌数据。每个块可能仅包含整个上下文令牌的一部分。例如，如果块大小为 16 且头大小为 128，则一个头的一个块可以存储 16 * 128 = 2048 个元素。
- **线程束（Warp）**：线程束是一组 32 个线程（`WARP_SIZE`），在流多处理器（SM）上同时执行。在该内核中，每个线程束一次处理一个查询令牌与一个块中所有键令牌的计算（可能通过多次迭代处理多个块）。例如，如果有 4 个线程束和 6 个上下文块，分配方式可能是线程束 0 处理第 0 和第 4 块，线程束 1 处理第 1 和第 5 块，线程束 2 处理第 2 块，线程束 3 处理第 3 块。
- **线程块**：线程块是一组线程（`NUM_THREADS`），可以访问相同的共享内存。每个线程块包含多个线程束（`NUM_WARPS`），在该内核中，每个线程块处理一个查询令牌与整个上下文的键令牌的计算。
- **网格（Grid）**：网格是线程块的集合，并定义了集合的形状。在该内核中，形状为 `(num_heads, num_seqs, max_num_partitions)`。因此，每个线程块仅处理一个头、一个序列和一个分区的计算。

## 查询

本节将介绍查询数据如何在内存中存储以及每个线程如何获取。如上所述，每个线程组获取一个查询令牌数据，而每个线程本身仅处理一个查询令牌数据的一部分。在每个线程束内，每个线程组将获取相同的查询令牌数据，但会与不同的键令牌数据进行乘法运算。

```cpp
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

<figure markdown="span">
  ![](../../assets/kernel/query.png){ align="center" alt="query" width="70%" }
</figure>

每个线程定义自己的 `q_ptr`，指向全局内存中分配的查询令牌数据。例如，如果 `VEC_SIZE` 为 4 且 `HEAD_SIZE` 为 128，则 `q_ptr` 指向的数据包含总共 128 个元素，分为 128 / 4 = 32 个向量。

<figure markdown="span">
  ![](../../assets/kernel/q_vecs.png){ align="center" alt="q_vecs" width="70%" }
</figure>

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

接下来，我们需要将 `q_ptr` 指向的全局内存数据读取到共享内存中作为 `q_vecs`。重要的是要注意，每个向量被分配到不同的行。例如，如果 `THREAD_GROUP_SIZE` 为 2，线程 0 将处理第 0 行的向量，而线程 1 将处理第 1 行的向量。通过这种方式读取查询数据，相邻线程（如线程 0 和线程 1）可以读取相邻内存，实现内存合并以提高性能。

## 键

与“查询”部分类似，本节介绍键的内存布局和分配。虽然每个线程组在一次内核运行中仅处理一个查询令牌，但它可能在多次迭代中处理多个键令牌。同时，每个线程束将在多次迭代中处理多个键令牌块，确保整个线程组在内核运行后处理所有上下文令牌。在本上下文中，“处理”指的是执行查询数据与键数据的点乘。

```cpp
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                    + kv_head_idx * kv_head_stride
                    + physical_block_offset * x;
```

与 `q_ptr` 不同，`k_ptr` 在每个线程中将在不同迭代中指向不同的键令牌。如上所示，`k_ptr` 根据 `k_cache` 指向分配的块、头和令牌的键令牌数据。

<figure markdown="span">
  ![](../../assets/kernel/key.png){ align="center" alt="key" width="70%" }
</figure>

上图展示了键数据的内存布局。假设 `BLOCK_SIZE` 为 16，`HEAD_SIZE` 为 128，`x` 为 8，`THREAD_GROUP_SIZE` 为 2，总共有 4 个线程束。每个矩形表示一个头中一个键令牌的所有元素，将由一个线程组处理。左侧显示线程束 0 的总共 16 个键令牌数据块，而右侧表示其他线程束或迭代的剩余键令牌数据。每个矩形内有总共 32 个向量（一个令牌的 128 个元素），将由 2 个线程（一个线程组）分别处理。

<figure markdown="span">
  ![](../../assets/kernel/k_vecs.png){ align="center" alt="k_vecs" width="70%" }
</figure>

```cpp
K_vec k_vecs[NUM_VECS_PER_THREAD]
```

接下来，我们需要从 `k_ptr` 读取键令牌数据并将其存储在寄存器内存中作为 `k_vecs`。我们使用寄存器内存存储 `k_vecs`，因为它仅被一个线程访问一次，而 `q_vecs` 会被多个线程多次访问。每个 `k_vecs` 将包含多个用于后续计算的向量。每个向量将在每次内层迭代中设置。向量的分配允许线程束中的相邻线程一起读取相邻内存，再次促进内存合并。例如，线程 0 将读取向量 0，而线程 1 将读取向量 1。在下一次内层循环中，线程 0 将读取向量 2，而线程 1 将读取向量 3，依此类推。

如果你对整体流程仍有些困惑，不用担心，请继续阅读下一节“QK”。它将以更清晰、更高级的方式说明查询和键的计算流程。

## QK

如下伪代码所示，在整个 for 循环块之前，我们获取一个令牌的查询数据并将其存储在 `q_vecs` 中。然后，在外层 for 循环中，我们迭代不同的 `k_ptrs`，它们指向不同的令牌，并在内层 for 循环中准备 `k_vecs`。最后，我们执行 `q_vecs` 与每个 `k_vecs` 的点乘。

```cpp
q_vecs = ...
for ... {
    k_ptr = ...
    for ... {
        k_vecs[i] = ...
    }
    ...
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
}
```

如前所述，每个线程一次仅获取部分查询和键令牌数据。然而，在 `Qk_dot<>::dot` 中会发生跨线程组的归约。因此，这里返回的 `qk` 不仅是部分查询和键令牌点乘的结果，而是整个查询和键令牌数据的完整结果。

例如，如果 `HEAD_SIZE` 的值为 128 且 `THREAD_GROUP_SIZE` 为 2，则每个线程的 `k_vecs` 将包含总共 64 个元素。然而，返回的 `qk` 实际上是 128 个查询元素与 128 个键元素点乘的结果。如果想了解点乘和归约的更多细节，可以参考 `Qk_dot<>::dot` 的实现。然而，为简单起见，我在此文档中不作详细介绍。

## Softmax

接下来，我们需要计算所有 `qk` 的归一化 softmax，如上所示，其中每个 $x$ 表示一个 `qk`。为此，我们必须获得所有 `qk` 的归约值 `qk_max`（$m(x)$）和 `exp_sum`（$\ell(x)$）。归约应在整个线程块中进行，涵盖查询令牌与所有上下文键令牌之间的结果。

$$
\begin{gather*}
m(x):=\max _i \quad x_i \\ \quad f(x):=\left[\begin{array}{lll}e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}\end{array}\right]\\ \quad \ell(x):=\sum_i f(x)_i \\
\quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}
\end{gather*}
$$

### `qk_max` 和 `logits`

在获得 `qk` 结果后，我们可以用 `qk` 设置临时 `logits` 结果（最终，`logits` 应存储归一化的 softmax 结果）。同时，我们可以比较并收集当前线程组计算的所有 `qk` 的 `qk_max`。

```cpp
if (thread_group_offset == 0) {
    const bool mask = token_idx >= context_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```

请注意，这里的 `logits` 在共享内存中，因此每个线程组将为其分配的上下文令牌设置字段。总体而言，`logits` 的大小应为上下文令牌的数量。

```cpp
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

if (lane == 0) {
    red_smem[warp_idx] = qk_max;
}
```

然后，我们需要获得每个线程束的归约 `qk_max`。主要思想是让线程束中的线程相互通信，获取最终的最大 `qk`。

```cpp
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

最后，我们可以通过比较线程块中所有线程束的 `qk_max` 来获得整个线程块的归约 `qk_max`。然后，我们需要将最终结果广播到每个线程。

### `exp_sum`

与 `qk_max` 类似，我们也需要获得整个线程块的归约和值。

```cpp
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}
...
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```

首先，求和每个线程组的所有 exp 值，同时将 `logits` 的每个条目从 `qk` 转换为 `exp(qk - qk_max)`。请注意，这里的 `qk_max` 已经是整个线程块的最大 `qk`。然后，我们可以像 `qk_max` 一样对 `exp_sum` 进行整个线程块的归约。

```cpp
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
```

最后，利用归约的 `qk_max` 和 `exp_sum`，我们可以获得最终的归一化 softmax 结果作为 `logits`。此 `logits` 变量将在后续步骤中用于与值数据的点乘。现在，它应存储所有分配的上下文令牌的 `qk` 的归一化 softmax 结果。

## 值

<figure markdown="span">
  ![](../../assets/kernel/value.png){ align="center" alt="value" width="70%" }
</figure>

<figure markdown="span">
  ![](../../assets/kernel/logits_vec.png){ align="center" alt="logits_vec" width="50%" }
</figure>

<figure markdown="span">
  ![](../../assets/kernel/v_vec.png){ align="center" alt="v_vec" width="70%" }
</figure>

现在我们需要检索值数据并与 `logits` 进行点乘。与查询和键不同，值数据没有线程组的概念。如图所示，与键令牌内存布局不同，同一列的元素对应于同一个值令牌。对于一个值数据块，有 `HEAD_SIZE` 行和 `BLOCK_SIZE` 列，被分成多个 `v_vecs`。

每个线程总是从同一 `V_VEC_SIZE` 个令牌中获取 `V_VEC_SIZE` 个元素。因此，单个线程通过多次内层迭代从不同行和相同列获取多个 `v_vecs`。对于每个 `v_vec`，需要与对应的 `logits_vec` 进行点乘，`logits_vec` 也是来自 `logits` 的 `V_VEC_SIZE` 个元素。总体而言，通过多次内层迭代，每个线程束将处理一个值令牌块。通过多次外层迭代，整个上下文值令牌被处理。

```cpp
float accs[NUM_ROWS_PER_THREAD];
for ... { // 不同块的迭代
    logits_vec = ...
    for ... { // 不同行的迭代
        v_vec = ...
        ...
        accs[i] += dot(logits_vec, v_vec);
    }
}
```

如上伪代码所示，在外层循环中，与 `k_ptr` 类似，`logits_vec` 迭代不同块并从 `logits` 读取 `V_VEC_SIZE` 个元素。在内层循环中，每个线程从相同令牌读取 `V_VEC_SIZE` 个元素作为 `v_vec` 并执行点乘。重要的是要注意，在每次内层迭代中，线程为相同令牌获取不同的头位置元素。点乘结果随后累积在 `accs` 中。因此，`accs` 的每个条目映射到当前线程分配的头位置。

例如，如果 `BLOCK_SIZE` 为 16 且 `V_VEC_SIZE` 为 8，则每个线程一次获取 8 个令牌的 8 个值元素。每个元素来自不同令牌的同一头位置。如果 `HEAD_SIZE` 为 128 且 `WARP_SIZE` 为 32，则每次内层循环，一个线程束需要获取 `WARP_SIZE * V_VEC_SIZE = 256` 个元素。这意味着一个线程束处理整个值令牌块需要 128 * 16 / 256 = 8 次内层迭代。每个线程中的 `accs` 包含 8 个元素，这些元素在 8 个不同头位置从所有分配的 8 个令牌累积。对于线程 0，`accs` 变量将有 8 个元素，分别是值头的第 0、32、…、224 个元素，从所有分配的 8 个令牌累积。

## LV

现在，我们需要在每个线程束内对 `accs` 进行归约。这个过程允许每个线程累积分配的头位置的所有令牌的 `accs`。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
}
```

接下来，我们对所有线程束的 `accs` 进行归约，允许每个线程拥有所有上下文令牌的分配头位置的 `accs` 累积。请注意，每个线程中的每个 `accs` 仅存储整个头的一部分元素的累积。然而，总体而言，所有输出结果都已计算，只是存储在不同的线程寄存器内存中。

```cpp
float* out_smem = reinterpret_cast<float*>(shared_mem);
for (int i = NUM_WARPS; i > 1; i /= 2) {
    // 上半部分线程束写入共享内存
    ...
    float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        ...
        dst[row_idx] = accs[i];
    }

    // 下半部分线程束更新输出
    const float* src = &out_smem[warp_idx * HEAD_SIZE];
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        ...
        accs[i] += src[row_idx];
    }

    // 写出 accs
}
```

## 输出

现在我们可以将所有计算结果从本地寄存器内存写入最终输出全局内存。

```cpp
scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                + head_idx * max_num_partitions * HEAD_SIZE
                + partition_idx * HEAD_SIZE;
```

首先，我们需要定义 `out_ptr` 变量，指向分配的序列和头的起始地址。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
    }
}
```

最后，我们需要迭代不同的分配头位置，并根据 `out_ptr` 写出相应的累积结果。