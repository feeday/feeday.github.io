---
layout:     post
title:      文件重命名
subtitle:   Python files Rename
date:       2024-03-09
author:     Puck
catalog: true
tags:
    - Python

---


## 文件重命名

通过Python脚本批量重命名指定后缀名文件

如果你想使用`tqdm`库，你可以通过以下命令使用pip进行安装：

```
pip install tqdm
```

此外，`os`和`glob`是Python的内置库，无需额外安装。

## 保存为 .py 后缀运行
```
import os
import glob
from tqdm import tqdm

# 文件扩展名和name变量定义
ext = ".png"
name = "day"
day = "20240304"

# 使用name变量值格式化文件夹路径
folder_path = r"d:\1" 
# 获取文件夹中的文件，并按创建时间升序排序
files = glob.glob(os.path.join(folder_path, "*" + ext))
files.sort(key=lambda x: os.path.getmtime(x))

# 创建进度条
progress_bar = tqdm(total=len(files), ncols=80)

# 遍历文件夹中的文件
for count, old_file_path in enumerate(files, start=1):
    # 构建新的文件名
    new_file_name = f"edge_{name}_{day}_{str(count).zfill(4)}{ext}"
    
    # 构建新文件的完整路径
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 当文件已存在时，重新命名文件
    while os.path.exists(new_file_path):
        count += 1
        new_file_name = f"edge_{name}_{day}_{str(count).zfill(4)}{ext}"
        new_file_path = os.path.join(folder_path, new_file_name)
    
    # 重命名文件
    os.rename(old_file_path, new_file_path)
    
    # 更新进度条
    progress_bar.update(1)

# 关闭进度条
progress_bar.close()

```
##  编程语言

- [Python](https://www.python.org/downloads/)
- [Python-370](https://www.python.org/downloads/release/python-370/)
- [Windows-312](https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe)
- [Python-install](https://www.runoob.com/python/python-install.html)
- [Visual Studio Code](https://code.visualstudio.com/Download)
  
```
python -V  # 查看 Python 版本
pip -V  # 查看 pip 版本
pip list # 查看已安装的包
pip install package_name #安装包
python -m pip install --upgrade pip # 更新 pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # 更新清华源

python3.7 -m venv myenv  # 使用以下命令创建一个名为 myenv 的基于 Python 3.7 的虚拟环境
myenv\Scripts\activate # 激活虚拟环境。在 Windows 系统
source myenv/bin/activate # 在 macOS 或 Linux 系统
```
