# vLLM V1

**我们已开始逐步淘汰V0。请阅读[RFC #18571](https://github.com/vllm-project/vllm/issues/18571)了解更多详情。**

V1现已成为所有支持用例的默认设置，我们将逐步为计划支持的每个用例启用V1。请在[GitHub](https://github.com/vllm-project/vllm)或[vLLM Slack](https://inviter.co/vllm-slack)上分享您的任何反馈！

若要禁用V1，请设置环境变量为：`VLLM_USE_V1=0`，并在GitHub上提交一个问题说明原因！

## 为什么选择vLLM V1？

vLLM V0成功支持了广泛的模型和硬件，但随着新功能的独立开发，系统变得越来越复杂。这种复杂性使得整合新功能更加困难，并带来了技术债务，凸显了对更简洁统一设计的需求。

在V0的成功基础上，vLLM V1保留了V0中稳定且经过验证的组件（例如模型、GPU内核和实用工具）。与此同时，它对核心系统（包括调度器、KV缓存管理器、工作进程、采样器和API服务器）进行了重大重新架构，提供了更具凝聚力、可维护的框架，以更好地适应持续的增长和创新。

具体来说，V1旨在：

- 提供**简单、模块化且易于修改的代码库**。
- 确保**高性能**，几乎零CPU开销。
- **整合关键优化**，形成统一的架构。
- **零配置**，默认启用功能/优化。

我们观察到升级到V1核心引擎后，性能显著提升，特别是在长上下文场景中。请查看性能基准（待添加）。

欲了解更多详情，请查看vLLM V1博客文章[vLLM V1：vLLM核心架构的重大升级](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（发布于2025年1月27日）。

本动态用户指南列出了vLLM V1引入的一些**重要变化和限制**。团队一直在积极努力将V1设为默认引擎，因此本指南将随着更多功能的支持而不断更新。

### 支持概览
#### 硬件

| 硬件 | 状态                                   |
|----------|------------------------------------------|
| **NVIDIA** | <nobr>🚀 原生支持</nobr>         |
| **AMD**    | <nobr>🚧 开发中</nobr>           |
| **TPU**    | <nobr>🚧 开发中</nobr>           |
| **CPU**    | <nobr>🚧 开发中</nobr>           |

#### 功能/模型

| 功能/模型 | 状态 |
|-----------------|-----------------------------------------------------------------------------------|
| **前缀缓存**                    | <nobr>🚀 已优化</nobr>                                                        |
| **分块预填充**                  | <nobr>🚀 已优化</nobr>                                                        |
| **LoRA**                        | <nobr>🚀 已优化</nobr>                                                         |
| **对数概率计算**                | <nobr>🟢 功能正常</nobr>                                                        |
| **多模态模型**                  | <nobr>🟢 功能正常</nobr>                                                        |
| **FP8 KV缓存**                  | <nobr>🟢 在Hopper设备上功能正常 ([PR #15191](https://github.com/vllm-project/vllm/pull/15191))</nobr>|
| **推测解码**                    | <nobr>🚧 开发中 ([PR #13933](https://github.com/vllm-project/vllm/pull/13933))</nobr>|
| **前缀缓存下的提示对数概率**    | <nobr>🟡 计划中 ([RFC #13414](https://github.com/vllm-project/vllm/issues/13414))</nobr>|
| **结构化输出替代后端**          | <nobr>🟡 计划中</nobr>                                                           |
| **嵌入模型**                    | <nobr>🚧 开发中 ([PR #16188](https://github.com/vllm-project/vllm/pull/16188))</nobr> |
| **Mamba模型**                   | <nobr>🟡 计划中</nobr>                                                           |
| **编码器-解码器模型**           | <nobr>🟠 延迟支持</nobr>                                                           |
| **请求级结构化输出后端**        | <nobr>🔴 已废弃</nobr>                                                        |
| **best_of**                     | <nobr>🔴 已废弃 ([RFC #13361](https://github.com/vllm-project/vllm/issues/13361))</nobr>|
| **每请求逻辑处理器**            | <nobr>🔴 已废弃 ([RFC #13360](https://github.com/vllm-project/vllm/pull/13360))</nobr> |
| **GPU <> CPU KV缓存交换**       | <nobr>🔴 已废弃</nobr>                                                        |

- **🚀 已优化**：几乎完全优化，目前无进一步计划。
- **🟢 功能正常**：完全可用，正在进行优化。
- **🚧 开发中**：正在积极开发。
- **🟡 计划中**：计划未来实现（部分可能有开放的PR/RFC）。
- **🟠 延迟支持**：在V1中暂时搁置，但计划稍后重新引入。
- **🔴 已废弃**：V1暂无计划支持，除非有强烈需求。

**注意**：vLLM V1的统一调度器以简单字典（例如 `{request_id: num_tokens}`）的方式处理提示和输出令牌，为每个请求动态分配固定令牌预算，支持分块预填充、前缀缓存和推测解码等功能，无需严格区分预填充和解码阶段。

### 语义变化和已废弃功能

#### 对数概率

vLLM V1支持对数概率和提示对数概率。然而，与V0相比存在一些重要的语义差异：

**对数概率计算**

V1中的对数概率在从模型原始输出计算后立即返回（即在应用任何逻辑后处理（如温度缩放或惩罚调整）之前）。因此，返回的对数概率不反映采样期间使用的最终调整概率。

对带后采样调整的对数概率支持正在进行中，将在未来更新中添加。

**前缀缓存下的提示对数概率**

当前仅在通过`--no-enable-prefix-caching`关闭前缀缓存时支持提示对数概率。在未来版本中，提示对数概率将与前缀缓存兼容，但即使在命中前缀缓存的情况下，也会触发重新计算以恢复完整的提示对数概率。详情见[RFC #13414](https://github.com/vllm-project/vllm/issues/13414)。

#### 已废弃功能

作为vLLM V1重大架构重构的一部分，一些遗留功能已被废弃。

**采样功能**

- **best_of**：由于使用量有限，此功能已被废弃。详情见[RFC #13361](https://github.com/vllm-project/vllm/issues/13361)。
- **每请求逻辑处理器**：在V0中，用户可以传递自定义处理函数以按请求调整逻辑。在vLLM V1中，此功能已被废弃。相反，设计正转向支持**全局逻辑处理器**，团队正在为未来版本积极开发此功能。详情见[RFC #13360](https://github.com/vllm-project/vllm/pull/13360)。

**KV缓存功能**

- **GPU <> CPU KV缓存交换**：凭借新的简化核心架构，vLLM V1不再需要KV缓存交换来处理请求抢占。

**结构化输出功能**

- **请求级结构化输出后端**：已废弃，替代后端（outlines、guidance）及其后备机制正在开发中。

### 功能与模型支持进展

尽管我们已在vLLM V1中重新实现并部分优化了V0中的许多功能和模型，但一些功能的优化仍在进行中，其他功能仍未支持。

#### 待优化功能

这些功能已在vLLM V1中支持，但优化仍在进行中。

- **推测解码**：当前V1仅支持基于ngram的推测解码。后续将支持其他类型的推测解码（例如，见[PR #13933](https://github.com/vllm-project/vllm/pull/13933)）。我们将优先支持Eagle、MTP等，而不是基于草稿模型的推测解码。

- **多模态模型**：V1与V0几乎完全兼容，但尚不支持交错模态输入。有关即将推出的功能和优化状态，请参见[此处](https://github.com/orgs/vllm-project/projects/8)。

#### 待支持功能

- **结构化输出替代后端**：计划支持结构化输出替代后端（outlines、guidance）。V1当前仅支持`xgrammar:no_fallback`模式，即如果输出模式不受xgrammar支持，将报错。有关结构化输出的详情，请参见[此处](https://docs.vllm.ai/en/latest/features/structured_outputs.html)。

#### 待支持模型

vLLM V1当前不包括具有`SupportsV0Only`协议的模型架构，大多数属于以下类别。V1将最终添加对这些模型的支持。

**嵌入模型**  
初始支持将通过[PR #16188](https://github.com/vllm-project/vllm/pull/16188)提供。

随后，我们将考虑使用[隐藏状态处理器](https://github.com/vllm-project/vllm/issues/12249)，基于[全局逻辑处理器](https://github.com/vllm-project/vllm/pull/13360)，以在V1中启用同一引擎实例同时进行生成和嵌入。

**Mamba模型**  
使用选择性状态空间机制（而不是标准变换器注意力）的模型尚未支持（例如，`MambaForCausalLM`、`JambaForCausalLM`）。

**编码器-解码器模型**  
vLLM V1当前针对解码器专用变换器进行了优化。需要独立编码器和解码器之间交叉注意力的模型尚未支持（例如，`BartForConditionalGeneration`、`MllamaForConditionalGeneration`）。

有关支持模型的完整列表，请参见[支持模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)。