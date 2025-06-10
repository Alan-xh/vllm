# 基于人类反馈的强化学习

基于人类反馈的强化学习 (RLHF) 是一种利用人类生成的偏好数据对语言模型进行微调的技术，旨在使模型输出与期望行为保持一致。

vLLM 可用于生成 RLHF 的补全语句。最佳方法是使用 [TRL](https://github.com/huggingface/trl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 和 [verl](https://github.com/volcengine/verl) 等库。

如果您不想使用现有库，请参阅以下基本示例以开始使用：

- [训练和推理过程位于不同的 GPU 上（受 OpenRLHF 启发）](../examples/offline_inference/rlhf.md)
- [使用 Ray 将训练和推理过程共置在同一 GPU 上](../examples/offline_inference/rlhf_colocate.md)
- [使用 vLLM 执行 RLHF 的实用程序](../examples/offline_inference/rlhf_utils.md)