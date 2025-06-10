# Transformers 强化学习

Transformers 强化学习 (TRL) 是一个全栈库，提供一套工具，用于训练 Transformer 语言模型，方法包括监督微调 (SFT)、群组相对策略优化 (GRPO)、直接偏好优化 (DPO)、奖励建模等。该库与 🤗 Transformers 集成。

诸如 GRPO 或在线 DPO 之类的在线方法需要模型生成补全。vLLM 可用于生成这些补全！

有关更多信息，请参阅 TRL 文档中的指南 [用于在线方法快速生成的 vLLM](https://huggingface.co/docs/trl/main/en/speeding_up_training#vllm-for-fast-generation-in-online-methods)。

!!!信息
有关您可以为这些在线方法的配置提供的 `use_vllm` 标志的更多信息，请参阅：
- [`trl.GRPOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig.use_vllm)
- [`trl.OnlineDPOConfig.use_vllm`](https://huggingface.co/docs/trl/main/en/online_dpo_trainer#trl.OnlineDPOConfig.use_vllm)