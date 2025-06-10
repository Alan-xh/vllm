# vLLM的`torch.compile`集成

在vLLM的V1架构中，`torch.compile`默认启用，是框架的关键部分。本文档提供了一个简单的示例，展示如何理解`torch.compile`的使用。

在整个示例中，我们将使用V1运行一个常见的Llama模型，并开启调试级别日志以显示所有细节。使用的命令是 `VLLM_USE_V1=1 VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`。

## 编译缓存

在非常详细的日志中，我们可以看到：

```
INFO 03-07 03:06:55 [backends.py:409] 使用缓存目录：~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 用于vLLM的torch.compile
```

vLLM会综合考虑所有相关因素，并决定一个存储所有编译产物的目录。这意味着，在部署场景中，你可以直接复制整个`~/.cache/vllm/torch_compile_cache`目录，以节省大量的编译时间，从而加速vLLM实例的启动时间。

考虑的因素包括：

- 所有相关配置（参见[config.py](gh-file:vllm/config.py)中的`compute_hash`函数）
- PyTorch配置（参见[compiler_interface.py](gh-file:vllm/compilation/compiler_interface.py)中的`compute_hash`函数）
- 模型的前向函数及前向函数调用的相关函数（见下文）

综合考虑这些因素，通常可以保证缓存安全使用，不会导致任何意外行为。因此，缓存默认启用。如果你想调试编译过程，或者怀疑缓存导致了一些问题，可以通过设置环境变量`VLLM_DISABLE_COMPILE_CACHE=1`来禁用缓存。

vLLM的`torch.compile`集成的独特之处在于，我们保证在处理任何请求之前完成所有编译。不会因为请求触发新的编译。否则，引擎会在该请求上阻塞，响应时间会出现意外的波动。

## Python代码编译

在非常详细的日志中，我们可以看到：

```
DEBUG 03-07 03:06:52 [decorators.py:203] 开始编译函数 <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

DEBUG 03-07 03:06:54 [backends.py:370] 跟踪文件（用于编译缓存考虑）：
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

DEBUG 03-07 03:07:07 [backends.py:462] 计算图保存至 ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo转换后的代码保存至 ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
```

这是关于Python代码编译的内容，即通过Dynamo进行图捕获。它尝试跟踪`xxx/vllm/model_executor/models/llama.py:339`中的函数，即我们编译的模型的`forward`函数。在前向传播过程中，Dynamo还会调用并内联其他函数，如日志所示，包括来自`xxx/torch/nn/modules/module.py`的一些PyTorch函数（因为模块属性访问会触发函数调用），以及vLLM的一些通信/注意力/激活函数。所有跟踪的文件都会在决定使用哪个缓存目录时被考虑。这样，上述文件中任何代码更改都会导致编译缓存未命中，从而触发重新编译。

Dynamo编译的结果是一个新函数，存储在`~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py`中。通常，这个函数会从模块中解包张量，然后将其传递给跟踪的计算图。计算图存储在`~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py`中。

## 计算图处理

计算图为每个张量都有形状注释。输入包括输入ID、位置ID、模型的权重和缓冲区，输出是最终的隐藏状态。注意，语言模型头部投影和采样操作不在图中考虑。

计算图的大多数输入具有静态形状，因为它们是模型的权重和缓冲区，在模型生命周期内不会变化。只有输入ID和位置ID具有符号形状，即形状可能因批次而异。然而，它们共享相同的符号形状。也就是说，计算图中唯一变化的尺寸是批次大小（当前前向传播处理的令牌数）。

注意力操作比较复杂，需要与键值缓存交互，形状也较为复杂。幸运的是，注意力操作的输出形状与输入查询的形状相同。因此，我们将整个注意力操作封装成一个PyTorch自定义操作`torch.ops.vllm.unified_attention_with_output`，这样Dynamo不会尝试检查其内部操作。这样，尽管注意力操作复杂，我们仍然可以从Dynamo的角度捕获模型的完整计算图。

计算图进一步被`splitting_ops`（通常是注意力操作）分割成多个部分。因此，在`~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py`文件中，我们可以看到许多子模块，每个子模块是分割后的一部分图：

- 注意力操作本身是一个子模块。
- 从一个注意力操作到下一个注意力操作的计算图部分是一个子模块。

每个子模块可以通过其索引识别，并被单独处理。

## 计算图编译

在非常详细的日志中，我们还可以看到：

```
DEBUG 03-07 03:52:37 [backends.py:134] 将第0个图（形状None）存储为Inductor编译结果，句柄为 ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] 将第1个图（形状None）存储为Inductor编译结果，句柄为 ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] 将第15个图（形状None）存储为Inductor编译结果，句柄为 ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] 将第16个图（形状None）存储为Inductor编译结果，句柄为 ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

这意味着第一个计算图片段（符号形状为`None`）由Inductor编译（键为`fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`）。编译后的内核存储在`~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py`中。你可以打开该文件查看Inductor最终运行的代码。

还有一个细节：你可以看到第1个图和第15个图具有相同的键，而第0个图和第16个图不同。这是预期的，因为我们按注意力操作分割图，得到3个独特的子图：

- 第一个注意力操作之前的层
- 从一个注意力操作到下一个注意力操作的中间层
- 最后一个注意力操作之后的层

如果我们已经有了缓存目录（例如第二次运行相同的代码），我们会看到以下日志：

```
DEBUG 03-07 04:00:45 [backends.py:86] 直接从Inductor加载第0个图（形状None），句柄为 ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

这次，Inductor编译完全被绕过，我们将从磁盘加载上次获得的编译产物。

上述示例仅使用Inductor为通用形状（即符号形状）进行编译。我们还可以使用Inductor为特定形状编译，例如：

```
vllm serve meta-llama/Llama-3.2-1B --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

然后，它还会为批次大小`1, 2, 4, 8`编译特定的内核。此时，计算图中的所有形状都是静态且已知的，我们会开启自动调优以获得最佳性能。第一次运行时可能较慢，但下次运行时，我们可以直接绕过调优并运行调优后的内核。

当所有形状已知时，`torch.compile`可以比较不同的配置，通常能找到更好的配置来运行内核。例如，我们可以看到以下日志：

```
AUTOTUNE mm(8x2048, 2048x3072)
  triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
  triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
  triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
  mm 0.0160 ms 81.6% 
  triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
  triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
  triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
  triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
  triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
  triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
```

这意味着，对于形状为`8x2048x3072`的矩阵乘法，`torch.compile`尝试了各种配置的Triton模板，速度比默认代码（分派到cublas库）快得多。

不幸的是，由于自动调优耗时较长（从几秒到几分钟，取决于模型大小和批次大小），尽管可以缓存以供后续使用，但为了用户友好性，我们默认关闭了它。如果想获得最大性能，建议尝试通过编译特定形状启用它。

## Cudagraph捕获

vLLM的V1架构使用分段cudagraph。完整计算图如上所述被分割，我们仅对注意力操作之间的图部分（包括第一个注意力操作前的图和最后一个注意力操作后的图）捕获cudagraph。这是基于一个常见观察：注意力操作之间的计算通常是按令牌进行的，易于cudagraph处理；而注意力操作本身不易与cudagraph兼容。因此，通过在eager模式下运行注意力操作，而在cudagraph中运行其余操作，我们保持了注意力操作的灵活性。

分段cudagraph还具有细粒度的内存管理。目的是仅将注意力内核排除在cudagraph之外，同时将所有其他模块和内存分配操作保留在cudagraph中。这就是为什么V1中的注意力操作将输出张量作为输入的原因。

cudagraph由编译器后端捕获和管理，并在批次大小有对应的cudagraph捕获时重放。模型的调用者（模型运行器）只需确保正确管理输入缓冲区。所有中间缓冲区由编译器后端自动管理。

默认情况下，vLLM会尝试确定一组捕获cudagraph的大小。你也可以通过配置`cudagraph_capture_sizes`覆盖它：

```
vllm serve meta-llama/Llama-3.2-1B --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

然后，它将仅为指定的大小捕获cudagraph。这对于cudagraph捕获的细粒度控制非常有用。

### 完整Cudagraph捕获

如果使用与cudagraph兼容的注意力后端，可以将注意力操作纳入cudagraph。这在某些情况下可以提高性能，例如小型模型的解码速度。使用`--compilation-config '{"full_cuda_graph": true}'`启用此功能。

目前只有FlashAttention 3兼容，且仅在禁用级联注意力时有效。