---
title: 解耦预填充（实验性）
---
[](){ #disagg-prefill }

本页面为您介绍 vLLM 中的解耦预填充功能。

!!! note
    此功能为实验性功能，可能会发生变化。

## 为什么需要解耦预填充？

主要有以下两个原因：

- **单独调整首 token 时间 (TTFT) 和 token 间延迟 (ITL)**。解耦预填充将大语言模型推理的预填充和解码阶段置于不同的 vLLM 实例中。这为您提供了灵活性，可以分配不同的并行策略（例如 `tp` 和 `pp`）来调整 TTFT 而不影响 ITL，或者调整 ITL 而不影响 TTFT。
- **控制尾部 ITL**。如果不使用解耦预填充，vLLM 可能会在一个请求的解码过程中插入一些预填充任务，从而导致较高的尾部延迟。解耦预填充可以帮助您解决此问题并控制尾部 ITL。分块预填充通过适当的分块大小也可以实现相同的目标，但在实践中很难确定正确的分块大小值。因此，解耦预填充是控制尾部 ITL 更可靠的方式。

!!! note
    解耦预填充不会提高吞吐量。

## 使用示例

请参阅 <gh-file:examples/online_serving/disaggregated_prefill.sh> 以获取解耦预填充的使用示例。

## 基准测试

请参阅 <gh-file:benchmarks/disagg_benchmarks> 以获取解耦预填充的基准测试。

## 开发

我们通过运行两个 vLLM 实例实现了解耦预填充：一个用于预填充（我们称之为预填充实例），另一个用于解码（我们称之为解码实例），然后使用连接器将预填充的 KV 缓存和结果从预填充实例传输到解码实例。

所有解耦预填充的实现代码位于 `vllm/distributed/kv_transfer` 下。

解耦预填充的关键抽象：

- **连接器（Connector）**：连接器允许 **KV 消费者** 从 **KV 生产者** 获取一批请求的 KV 缓存。
- **查找缓冲区（LookupBuffer）**：查找缓冲区提供两个 API：`insert` KV 缓存和 `drop_select` KV 缓存。`insert` 和 `drop_select` 的语义类似于 SQL，其中 `insert` 将 KV 缓存插入缓冲区，`drop_select` 返回符合给定条件的 KV 缓存并将其从缓冲区中删除。
- **管道（Pipe）**：用于张量传输的单向 FIFO 管道。它支持 `send_tensor` 和 `recv_tensor`。

!!! note
    `insert` 是非阻塞操作，但 `drop_select` 是阻塞操作。

以下是展示上述三个抽象如何组织的图表：

![解耦预填充抽象](../assets/features/disagg_prefill/abstraction.jpg)

解耦预填充的工作流程如下：

![解耦预填充工作流程](../assets/features/disagg_prefill/overview.jpg)

`buffer` 对应于 LookupBuffer 中的 `insert` API，`drop_select` 对应于 LookupBuffer 中的 `drop_select` API。

## 第三方贡献

解耦预填充与基础设施高度相关，因此 vLLM 依赖第三方连接器来实现生产级别的解耦预填充（vLLM 团队将积极审查并合并新的第三方连接器 PR）。

我们推荐以下三种实现方式：

- **完全定制化连接器**：实现您自己的 `Connector`，并调用第三方库来发送和接收 KV 缓存，以及更多功能（例如编辑 vLLM 的模型输入以执行定制化预填充等）。这种方法为您提供最大的控制权，但存在与未来 vLLM 版本不兼容的风险。
- **类数据库连接器**：实现您自己的 `LookupBuffer`，并支持类似于 SQL 的 `insert` 和 `drop_select` API。
- **分布式 P2P 连接器**：实现您自己的 `Pipe`，并支持类似于 `torch.distributed` 的 `send_tensor` 和 `recv_tensor` API。