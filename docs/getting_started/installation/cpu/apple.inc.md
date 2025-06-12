# --8<-- [start:installation]

vLLM 对 macOS 的 Apple 芯片提供了实验性支持。目前，用户需要从源代码构建 vLLM 以在 macOS 上原生运行。

当前 macOS 的 CPU 实现支持 FP32 和 FP16 数据类型。

!!! warning
    该设备没有预构建的轮子或镜像，因此必须从源代码构建 vLLM。

# --8<-- [end:installation]
# --8<-- [start:requirements]

- 操作系统：`macOS Sonoma` 或更高版本
- SDK：`XCode 15.4` 或更高版本，包含命令行工具
- 编译器：`Apple Clang >= 15.0.0`

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

在安装 XCode 和包含 Apple Clang 的命令行工具后，执行以下命令从源代码构建并安装 vLLM。

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements/cpu.txt
pip install -e . 
```

!!! note
    在 macOS 上，`VLLM_TARGET_DEVICE` 自动设置为 `cpu`，目前这是唯一支持的设备。

#### 故障排除

如果构建过程中出现如下错误片段，提示无法找到标准 C++ 头文件，请尝试卸载并重新安装您的
[XCode 命令行工具](https://developer.apple.com/download/all/)。

```text
[...] fatal error: 'map' file not found
          1 | #include <map>
            |          ^~~~~
      1 error generated.
      [2/8] Building CXX object CMakeFiles/_C.dir/csrc/cpu/pos_encoding.cpp.o

[...] fatal error: 'cstddef' file not found
         10 | #include <cstddef>
            |          ^~~~~~~~~
      1 error generated.
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]