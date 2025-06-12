# 欢迎体验 vLLM

<p align="center">
  <img src="assets/logos/vllm-logo-only-light.ico" alt="vLLM Logo" style="display: block; margin: 0 auto; max-width: 60%; height: auto;">
</p>

<p style="text-align:center">
<strong>为所有人提供简单、快速且低成本的 LLM 服务
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">星标</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">关注</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">分叉</a>
</p>

vLLM 是一个快速且易用的 LLM 推理和服务库。

vLLM 最初由 [加州伯克利 Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，现已演变为一个由学术界和工业界共同贡献的社区驱动项目。

vLLM 的高效特性包括：

- 最先进的吞吐量
- 通过 [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) 高效管理注意力键和值内存
- 持续批处理输入请求
- 使用 CUDA/HIP 图加速模型执行
- 量化支持：[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、INT4、INT8 和 FP8
- 优化的 CUDA 内核，集成 FlashAttention 和 FlashInfer
- 推测解码
- 分块预填充

vLLM 的灵活性和易用性体现在：

- 与热门 HuggingFace 模型无缝集成
- 支持多种解码算法的高吞吐量服务，包括*并行采样*、*束搜索*等
- 支持张量并行和流水线并行以进行分布式推理
- 流式输出
- 兼容 OpenAI 的 API 服务器
- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU、Gaudi® 加速器和 GPU、IBM Power CPU、TPU，以及 AWS Trainium 和 Inferentia 加速器
- 前缀缓存支持
- 多 LoRA 支持

了解更多信息，请查看以下内容：

- [vLLM 发布博客文章](https://vllm.ai)（PagedAttention 简介）
- [vLLM 论文](https://arxiv.org/abs/2309.06180)（SOSP 2023）
- [如何通过持续批处理实现 LLM 推理 23 倍吞吐量并降低 p50 延迟](https://www.anyscale.com/blog/continuous-batching-llm-inference) 作者：Cade Daniel 等
- [vLLM 见面会][meetups]