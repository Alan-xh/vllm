---
title: 自动前缀缓存
---
[](){ #automatic-prefix-caching }

## 引言

自动前缀缓存（简称 APC）会缓存已有查询的键值（KV）缓存，使得新查询如果与已有查询共享相同前缀，可以直接重用该 KV 缓存，从而跳过共享部分的计算。

!!! note
    有关 vLLM 如何实现 APC 的技术细节，请参见[此处][design-automatic-prefix-caching]。

## 在 vLLM 中启用 APC

在 vLLM 引擎中设置 `enable_prefix_caching=True` 以启用 APC。以下是一个示例：

<gh-file:examples/offline_inference/automatic_prefix_caching.py>

## 示例工作负载

我们描述了两个示例工作负载，在这些场景中 APC 能够带来显著的性能提升：

- 长文档查询，用户反复查询同一长文档（例如软件手册或年度报告）但使用不同的查询。在这种情况下，APC 使 vLLM 仅需处理长文档*一次*，而所有后续请求都可以通过重用其 KV 缓存避免重新计算长文档。这使得 vLLM 能够以更高的吞吐量和更低的延迟处理后续请求。
- 多轮对话，用户可能在同一聊天会话中与应用程序进行多次交互。在这种情况下，APC 允许 vLLM 重用聊天历史的处理结果，跨所有后续对话轮次重用，从而使 vLLM 能够以更高的吞吐量和更低的延迟处理后续请求。

## 限制

一般来说，APC 不会降低 vLLM 的性能。尽管如此，APC 仅减少处理查询的时间（预填充阶段），而不会减少生成新 token 的时间（解码阶段）。因此，当 vLLM 大部分时间用于生成查询的回答（例如回答内容较长时），或新查询与任何已有查询不共享相同前缀（无法重用计算）时，APC 不会带来性能提升。