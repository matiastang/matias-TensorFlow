# tensorflow安装

[mac安装TensorFlow](https://github.com/xitu/tensorflow-docs/blob/zh-hans/install/install_mac.md)

## 通过Virtualenv 安装

1. 终端安装`pip`安装了跳过该步骤：
```
sudo easy_install pip
```

2. 安装`Virtualenv`：
```
pip install --upgrade virtualenv
```

3. 创建 Virtualenv 环境：
```
virtualenv --system-site-packages targetDirectory # 对应 Python 2.7
virtualenv --system-site-packages -p python3 targetDirectory # 对应 Python 3.n
```
其中 targetDirectory 表示 Virtualenv 目录树所在的顶层路径。我们假设 targetDirectory 为 ~/tensorflow，但你也可以选择任何你喜欢的路径。

4. 激活 Virtualenv 环境：
```
cd targetDirectory
source ./bin/activate # 如果是使用 bash、sh、ksh、或 zsh
source ./bin/activate.csh # 如果是使用 csh 或 tcsh 
```
前面的 source 命令会将你的命令行提示更改为以下内容：
```
(targetDirectory)$
```

5. 确保安装的 pip 版本大于或等于 8.1：
```
(targetDirectory)$ easy_install -U pip
```

6. 执行下面的命令会将 TensorFlow 及其全部依赖安装至 Virtualenv 环境中：
```
(targetDirectory)$ pip install --upgrade tensorflow      # 对应 Python 2.7
(targetDirectory)$ pip3 install --upgrade tensorflow     # 对应 Python 3.n 
```

7. 可选）如果第 6 步失败了（通常可能是因为你使用的 pip 版本小于 8.1），你还可以在激活的 Virtualenv 环境下，通过下面的命令安装 TensorFlow：
```
pip install --upgrade tfBinaryURL # Python 2.7
pip3 install --upgrade tfBinaryURL # Python 3.n 
```
其中 tfBinaryURL 指向 TensorFlow Python 软件包所在的 URL。合适的 tfBinaryURL 取决于你的操作系统和 Python 版本。你可以在[这儿](https://github.com/xitu/tensorflow-docs/blob/zh-hans/install/install_mac.md#the_url_of_the_tensorflow_python_package)找到你系统所对应的 tfBinaryURL。例如，如果你要在安装了 Python 2.7 的 macOS 上安装 TensorFlow，那么可以执行下面的命令：
```
pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl
```