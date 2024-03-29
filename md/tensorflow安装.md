<!--
 * @Author: matiastang
 * @Date: 2022-07-26 16:33:46
 * @LastEditors: matiastang
 * @LastEditTime: 2022-08-01 14:49:36
 * @FilePath: /matias-TensorFlow/md/tensorflow安装.md
 * @Description: Tensorflow安装
-->
# tensorflow安装

[mac安装TensorFlow](https://github.com/xitu/tensorflow-docs/blob/zh-hans/install/install_mac.md)
[版本支持及下载](https://pypi.org/project/tensorflow/#files)

# M1安装

目前anconda官网上对于macos系统只支持X86结构的，因此基于arm结构的MacOS暂时无法使用anconda来对tensorflow进行配置。虽然Anconda无法在M1上运行，但是相应的替代品有Miniforge，同样也能实现类似的效果。此外，苹果官方称还支持添加metal插件，从而可以调用集成在M1芯片中的GPU部分，但是目前只支持tensorflow2.5和2.6的版本，而本节使用使用的版本是tensorflow2.4 python3.8，因此并未安装加速插件。官方安装metal方法如下：

[Tensorflow Plugin - Metal - Apple Developer](https://developer.apple.com/metal/tensorflow-plugin/)

通解是安装miniforge的arm64版本并在这种conda里安装

[安装参考](https://zhuanlan.zhihu.com/p/478696535)
[Anaconda和miniforge](https://zhuanlan.zhihu.com/p/379567315)
[Anaconda和miniforge区别](https://qa.1r1g.com/sf/ask/4237287491/)
[miniforge官网](https://github.com/conda-forge/miniforge)

## 通过Virtualenv 安装

1. 终端安装`pip`安装了跳过该步骤：
```
sudo easy_install pip
```

2. 安装`Virtualenv`：
```
pip install --upgrade virtualenv
pip3 install --upgrade virtualenv
```

3. 创建 Virtualenv 环境：
```
virtualenv --system-site-packages targetDirectory # 对应 Python 2.7
virtualenv --system-site-packages -p python3 targetDirectory # 对应 Python 3.n

mkdir tensorflowLearning
virtualenv --system-site-packages -p python3 ~/tensorflowLearning
created virtual environment CPython3.7.2.final.0-64 in 1030ms
  creator CPython3Posix(dest=/Users/yunxi/tensorflowLearning, clear=False, global=True)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/Users/yunxi/Library/Application Support/virtualenv)
    added seed packages: pip==20.1.1, setuptools==49.1.0, wheel==0.34.2
  activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator
```
其中 targetDirectory 表示 Virtualenv 目录树所在的顶层路径。我们假设 targetDirectory 为 ~/tensorflow，但你也可以选择任何你喜欢的路径。

4. 激活 Virtualenv 环境：
```
cd targetDirectory
source ./bin/activate # 如果是使用 bash、sh、ksh、或 zsh
source ./bin/activate.csh # 如果是使用 csh 或 tcsh 

╭─yunxi@zfqdeMac-mini.local ~
╰─➤  cd tensorflowLearning
╭─yunxi@zfqdeMac-mini.local ~/tensorflowLearning
╰─➤  source ./bin/activate
(tensorflowLearning) ╭─yunxi@zfqdeMac-mini.local ~/tensorflowLearning
╰─➤
```
前面的 source 命令会将你的命令行提示更改为以下内容：
```
(targetDirectory)$
```

5. 确保安装的 pip 版本大于或等于 8.1：
```
(targetDirectory)$ easy_install -U pip

pip -V
pip 20.1.1 from /Users/yunxi/tensorflowLearning/lib/python3.7/site-packages/pip (python 3.7)
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

## 问题

### `pipx install tensorflow`安装不成功

```
$ pipx install tensorflow
Fatal error from pip prevented installation. Full pip output in file:
    /Users/matias/.local/pipx/logs/cmd_2022-07-26_17.44.15_pip_errors.log

Some possibly relevant errors from pip install:
    ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
    ERROR: No matching distribution found for tensorflow

Error installing tensorflow.
```
`pip3`可以安装
```
$ pip3 install tensorflow
Defaulting to user installation because normal site-packages is not writeable
Collecting tensorflow
  Downloading tensorflow-2.9.1-cp38-cp38-macosx_10_14_x86_64.whl (228.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/228.5 MB 27.4 kB/s eta 2:18:15
```
`pipenv`报错
```
$ pipenv install tensorflow
Creating a virtualenv for this project...
Pipfile: /Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/Pipfile
Using /Users/matias/.pyenv/versions/3.10.5/bin/python3 (3.10.5) to create virtualenv...
⠴ Creating virtual environment...created virtual environment CPython3.10.5.final.0-64 in 376ms
  creator CPython3Posix(dest=/Users/matias/.local/share/virtualenvs/python-mb9SXkT4, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/Users/matias/Library/Application Support/virtualenv)
    added seed packages: pip==21.2.4, setuptools==58.0.4, wheel==0.37.0
  activators BashActivator,CShellActivator,FishActivator,NushellActivator,PowerShellActivator,PythonActivator

✔ Successfully created virtual environment! 
Virtualenv location: /Users/matias/.local/share/virtualenvs/python-mb9SXkT4
Creating a Pipfile for this project...
Installing tensorflow...
Error:  An error occurred while installing tensorflow!
Error text: 
ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
ERROR: No matching distribution found for tensorflow

✘ Installation Failed
```

### pip3 install tensorflow 太慢了

使用`pip3 install tensorflow -i https://pypi.douban.com/simple`下载，比较快。

### 21827 illegal hardware instruction  python3
`M1`报`21827 illegal hardware instruction  python3`的错误，大概的原因就是不兼容arm架构
[查看](https://stackoverflow.com/questions/65383338/zsh-illegal-hardware-instruction-python-when-installing-tensorflow-on-macbook)

下载`tensorflow`安装包（支持`arm架构版本`的
* 安装TensorFlow macos版
```
$ pip3 install tensorflow-macos -i https://pypi.douban.com/simple
```
[M1安装tensorflow](https://zhuanlan.zhihu.com/p/478696535)