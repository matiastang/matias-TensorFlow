<!--
 * @Author: matiastang
 * @Date: 2022-08-01 14:59:56
 * @LastEditors: matiastang
 * @LastEditTime: 2022-08-01 17:36:31
 * @FilePath: /matias-TensorFlow/md/miniforge安装.md
 * @Description: 
-->
# miniforge

[miniforge官网](https://github.com/conda-forge/miniforge)

## 安装miniforge

```
$ brew install miniforge
...
Linking Binary 'conda' to '/opt/homebrew/bin/conda'
🍺  miniforge was successfully installed!
```
**注意**安装完成后重启终端
此时就可以使用`conda insall`安装所需库了，比如`pandas`，输入`conda install pandas`就会帮你自动安装此库

## 更换镜像源

确实现在`miniforge`我们已安装成功，并能正常使用，但对于国内用户来讲，下载速度实在是太慢了，我们需更改其默认镜像源，比如我将其改为清华镜像源进行下载，那下载速度简直不要太快
首先打开终端，输入以下命令
```
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --set show_channel_urls yes
```
确认更改`conda help`可以看到`config`的地址，默认为
```
$ conda help
...
config       Modify configuration values in .condarc. This is modeled after the git config command. Writes to the user
                 .condarc file (/Users/matias/.condarc) by default.
...
```
```
$ cat ./condarc
```

## 创建虚拟环境

用`conda`创建一个虚拟环境，同时设置`python`版本
`conda create -n 虚拟环境名称 python=版本号`
`conda create -n tensorflow python=3.8`
```
$ conda create -n mt_tensorflow python=3.10
...
#
# To activate this environment, use
#
#     $ conda activate mt_tensorflow
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

## 激活虚拟环境

`conda activate 虚拟环境名`
`conda activate tensorflow`
**提示**如果需要取消激活状态，输入`conda deactivate`即可
```
$ conda activate mt_tensorflow

CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.
```
```
$ conda init
no change     /opt/homebrew/Caskroom/miniforge/base/condabin/conda
no change     /opt/homebrew/Caskroom/miniforge/base/bin/conda
no change     /opt/homebrew/Caskroom/miniforge/base/bin/conda-env
no change     /opt/homebrew/Caskroom/miniforge/base/bin/activate
no change     /opt/homebrew/Caskroom/miniforge/base/bin/deactivate
no change     /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
no change     /opt/homebrew/Caskroom/miniforge/base/etc/fish/conf.d/conda.fish
no change     /opt/homebrew/Caskroom/miniforge/base/shell/condabin/Conda.psm1
no change     /opt/homebrew/Caskroom/miniforge/base/shell/condabin/conda-hook.ps1
no change     /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/xontrib/conda.xsh
no change     /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.csh
modified      /Users/matias/.bash_profile

==> For changes to take effect, close and re-open your current shell. <==
```
`zsh`初始化
```
$ conda init zsh
no change     /opt/homebrew/Caskroom/miniforge/base/condabin/conda
no change     /opt/homebrew/Caskroom/miniforge/base/bin/conda
no change     /opt/homebrew/Caskroom/miniforge/base/bin/conda-env
no change     /opt/homebrew/Caskroom/miniforge/base/bin/activate
no change     /opt/homebrew/Caskroom/miniforge/base/bin/deactivate
no change     /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
no change     /opt/homebrew/Caskroom/miniforge/base/etc/fish/conf.d/conda.fish
no change     /opt/homebrew/Caskroom/miniforge/base/shell/condabin/Conda.psm1
no change     /opt/homebrew/Caskroom/miniforge/base/shell/condabin/conda-hook.ps1
no change     /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/xontrib/conda.xsh
no change     /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.csh
modified      /Users/matias/.zshrc

==> For changes to take effect, close and re-open your current shell. <==
```
激活虚拟环境，取消激活状态使用`conda deactivate`
```
(base)  ~  $ conda activate mt_tensorflow
(mt_tensorflow)  ~ 
```
```
(mt_tensorflow)  ~  which python3
/opt/homebrew/Caskroom/miniforge/base/envs/mt_tensorflow/bin/python3
(mt_tensorflow)  ~  which python
/opt/homebrew/Caskroom/miniforge/base/envs/mt_tensorflow/bin/python
(mt_tensorflow)  ~  which pip3
/opt/homebrew/Caskroom/miniforge/base/envs/mt_tensorflow/bin/pip3
(mt_tensorflow)  ~  which pip
/opt/homebrew/Caskroom/miniforge/base/envs/mt_tensorflow/bin/pip
(mt_tensorflow)  ~  pip3 install tensorflow-macos
Collecting tensorflow-macos
```
## 安装tensorflow

当创建完虚拟环境后，做完准备工作之后，我们需要安装tensorflow-macos，这是我们真正的目的。下载`tensorflow`安装包（支持arm架构版本的）
下载慢
```
$ pip3 install tensorflow-macos
```
使用下面的下载
```
pip3 install tensorflow-macos -i https://pypi.douban.com/simple
pip3 install tensorflow-metal -i https://pypi.douban.com/simple
```
1. 测试`tensorflow`
```
$ python3 /Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_install_test.py
2.9.2
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
2. VSCode配置
```py
import tensorflow as tf
# 编辑器提示：Import "tensorflow" could not be resolvedPylancereportMissingImports
```
下载了`tensorflow`还有上述提示，则说明你配置了多个`python`环境，而编辑器目前所在的`python`环境没有下载该包，可以选择更换编辑器环境或者重新在编辑器的环境下下载。
更换环境步骤：`ctrl+shift+p` -->输入：`python:select interpreter`选择下载了该包的环境。

* 在`conda`的虚拟环境中查看`python3`的地址：
```
(mt_tensorflow)  ~  which python3
/opt/homebrew/Caskroom/miniforge/base/envs/mt_tensorflow/bin/python3
```
点击`Enter interpreter Path`将上面地址添加进去