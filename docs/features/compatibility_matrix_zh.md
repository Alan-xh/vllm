---
title: 兼容性矩阵
---
[](){ #compatibility-matrix }

以下表格显示了互斥功能以及在某些硬件上的支持情况。

使用的符号含义如下：

- ✅ = 完全兼容
- 🟠 = 部分兼容
- ❌ = 不兼容
- ❔ = 未知或待定

!!! note
    请查看带有链接的 ❌ 或 🟠 以了解不支持的功能/硬件组合的跟踪问题。

## 功能 x 功能

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 功能 | [CP][chunked-prefill] | [APC][automatic-prefix-caching] | [LoRA][lora-adapter] | <abbr title="提示适配器">提示适配器</abbr> | [SD][spec-decode] | CUDA 图 | <abbr title="池化模型">池化</abbr> | <abbr title="编码器-解码器模型">编码-解码</abbr> | <abbr title="对数概率">logP</abbr> | <abbr title="提示对数概率">提示 logP</abbr> | <abbr title="异步输出处理">异步输出</abbr> | 多步 | <abbr title="多模态输入">多模态</abbr> | 最佳选择 | 束搜索 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [CP][chunked-prefill] | ✅ | | | | | | | | | | | | | | |
| [APC][automatic-prefix-caching] | ✅ | ✅ | | | | | | | | | | | | | |
| [LoRA][lora-adapter] | ✅ | ✅ | ✅ | | | | | | | | | | | | |
| <abbr title="提示适配器">提示适配器</abbr> | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | |
| [SD][spec-decode] | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | | | | |
| CUDA 图 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | |
| <abbr title="池化模型">池化</abbr> | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | | | | | | | | |
| <abbr title="编码器-解码器模型">编码-解码</abbr> | ❌ | [❌](gh-issue:7366) | ❌ | ❌ | [❌](gh-issue:7366) | ✅ | ✅ | ✅ | | | | | | | |
| <abbr title="对数概率">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | |
| <abbr title="提示对数概率">提示 logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | |
| <abbr title="异步输出处理">异步输出</abbr> | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | |
| 多步 | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | |
| <abbr title="多模态输入">多模态</abbr> | ✅ | [🟠](gh-pr:8348) | [🟠](gh-pr:4194) | ❔ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | |
| 最佳选择 | ✅ | ✅ | ✅ | ✅ | [❌](gh-issue:6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](gh-issue:7968) | ✅ | ✅ | |
| 束搜索 | ✅ | ✅ | ✅ | ✅ | [❌](gh-issue:6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](gh-issue:7968) | ❔ | ✅ | ✅ |

[](){ #feature-x-hardware }

## 功能 x 硬件

| 功能                                                   | Volta              | Turing   | Ampere   | Ada   | Hopper   | CPU                | AMD   |
|-----------------------------------------------------------|--------------------|----------|----------|-------|----------|--------------------|-------|
| [CP][chunked-prefill]                                     | [❌](gh-issue:2729) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| [APC][automatic-prefix-caching]                           | [❌](gh-issue:3687) | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| [LoRA][lora-adapter]                                      | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="提示适配器">提示适配器</abbr>           | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](gh-issue:8475) | ✅     |
| [SD][spec-decode]                                         | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| CUDA 图                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ✅     |
| <abbr title="池化模型">池化</abbr>               | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❔     |
| <abbr title="编码器-解码器模型">编码-解码</abbr>       | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ❌     |
| <abbr title="多模态输入">多模态</abbr>                 | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="对数概率">logP</abbr>                        | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="提示对数概率">提示 logP</abbr>           | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| <abbr title="异步输出处理">异步输出</abbr> | ✅                  | ✅        | ✅        | ✅     | ✅        | ❌                  | ❌     |
| 多步                                                | ✅                  | ✅        | ✅        | ✅     | ✅        | [❌](gh-issue:8477) | ✅     |
| 最佳选择                                                   | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |
| 束搜索                                               | ✅                  | ✅        | ✅        | ✅     | ✅        | ✅                  | ✅     |

!!! note
    请参阅 [通过 NxD 推理后端的功能支持][feature-support-through-nxd-inference-backend] 以了解 AWS Neuron 硬件上支持的功能。