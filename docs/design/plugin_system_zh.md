---
title: vLLM的插件系统
---
[](){ #plugin-system }

社区经常请求能够扩展vLLM以添加自定义功能。为了满足这一需求，vLLM包含了一个插件系统，允许用户在不修改vLLM代码库的情况下添加自定义功能。本文档解释了vLLM中插件的工作原理以及如何为vLLM创建插件。

## vLLM中插件的工作原理

插件是用户注册的代码，由vLLM执行。鉴于vLLM的架构（参见[Arch Overview][arch-overview]），可能会涉及多个进程，特别是在使用各种并行技术进行分布式推理时。为了成功启用插件，vLLM创建的每个进程都需要加载插件。这通过`vllm.plugins`模块中的[load_general_plugins](https://github.com/vllm-project/vllm/blob/c76ac49d266e27aa3fea84ef2df1f813d24c91c7/vllm/plugins/__init__.py#L16)函数实现。该函数在vLLM创建的每个进程开始任何工作之前被调用。

## vLLM如何发现插件

vLLM的插件系统使用标准的Python `entry_points`机制。该机制允许开发者在他们的Python包中注册函数，供其他包使用。以下是一个插件示例：

```python
# 在`setup.py`文件中
from setuptools import setup

setup(name='vllm_add_dummy_model',
      version='0.1',
      packages=['vllm_add_dummy_model'],
      entry_points={
          'vllm.general_plugins':
          ["register_dummy_model = vllm_add_dummy_model:register"]
      })

# 在`vllm_add_dummy_model.py`文件中
def register():
    from vllm import ModelRegistry

    if "MyLlava" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyLlava",
            "vllm_add_dummy_model.my_llava:MyLlava",
        )
```

有关在包中添加入口点的更多信息，请查看[官方文档](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)。

每个插件包含三个部分：

1. **插件组**：入口点组的名称。vLLM使用入口点组`vllm.general_plugins`来注册通用插件。这是`setup.py`文件中`entry_points`的键。对于vLLM的通用插件，始终使用`vllm.general_plugins`。
2. **插件名称**：插件的名称。这是`entry_points`字典中值的名称。在上述示例中，插件名称为`register_dummy_model`。可以通过`VLLM_PLUGINS`环境变量按名称过滤插件。要仅加载特定插件，请将`VLLM_PLUGINS`设置为插件名称。
3. **插件值**：在插件系统中注册的函数的完全限定名称。在上述示例中，插件值为`vllm_add_dummy_model:register`，它指的是`vllm_add_dummy_model`模块中的`register`函数。

## 支持的插件类型

- **通用插件**（组名称为`vllm.general_plugins`）：这些插件的主要用例是将自定义的、树外模型注册到vLLM中。这是通过在插件函数中调用`ModelRegistry.register_model`来注册模型实现的。

- **平台插件**（组名称为`vllm.platform_plugins`）：这些插件的主要用例是将自定义的、树外平台注册到vLLM中。插件函数在当前环境不支持该平台时应返回`None`，在支持该平台时返回平台的完全限定名称。

## 编写插件的指导原则

- **可重入性**：入口点中指定的函数应该是可重入的，即可以多次调用而不会引发问题。这是必要的，因为在某些进程中该函数可能会被多次调用。

## 兼容性保证

vLLM保证文档化的插件接口（例如`ModelRegistry.register_model`）将始终可用于插件注册模型。然而，插件开发者有责任确保他们的插件与目标vLLM版本兼容。例如，`"vllm_add_dummy_model.my_llava:MyLlava"`应与插件目标的vLLM版本兼容。模型的接口在vLLM开发过程中可能会发生变化。