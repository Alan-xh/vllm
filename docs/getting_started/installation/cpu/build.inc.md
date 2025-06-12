首先，安装推荐的编译器。我们建议使用 `gcc/g++ >= 12.3.0` 作为默认编译器，以避免潜在问题。例如，在 Ubuntu 22.4 上，你可以运行：

```console
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

其次，克隆 vLLM 项目：

```console
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

第三，安装 vLLM CPU 后端构建所需的 Python 包：

```console
pip install --upgrade pip
pip install "cmake>=3.26.1" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

最后，构建并安装 vLLM CPU 后端：

```console
VLLM_TARGET_DEVICE=cpu python setup.py install
```

如果你想开发 vLLM，可以改为以可编辑模式安装：

```console
VLLM_TARGET_DEVICE=cpu python setup.py develop
```