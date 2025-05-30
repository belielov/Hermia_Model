# Hermia_Model
## 打包为`exe`文件的步骤
1. conda创建虚拟环境
```python
conda create -n 虚拟环境名字 python==3.6  #创建虚拟环境
 
conda activate 虚拟环境名字  #激活虚拟环境
 
conda deactivate  #退出虚拟环境
```
2. 安装所需的库
```python
pip list  # 查看已经安装的库

pip install matplotlib  # 安装缺少的库（以 matplotlib 为例）
```
3. 执行打包操作
```python
python -m PyInstaller -F -w -i favicon.ico membrane_fitting_v2.py

python -m PyInstaller  # 基础命令
-F                     # 生成单个可执行文件
-w                     # 隐藏控制台窗口（适用于GUI程序）
-i favicon.ico         # 设置生成文件的图标
membrane_fitting_v2.py # 要打包的Python脚本
```
## 解决matplotlib中文和负号显示乱码问题
```python
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```
## 一些代码说明
### `x.strip()`方法
 `x.strip()`是 Python 中字符串方法，用于**清理字符串两端的特定字符**。
1. 默认行为（无参数）
- **作用**：去除字符串**开头和结尾**的所有**空白字符**（包括空格、制表符`\t`、换行符`\n`、回车符`\r`等）
- **示例**：
```python
text = " \tHello, World!\n "
print(text.strip())  # 输出："Hello, World!"
```
2. 指定去除字符（带参数）
- **作用**：若传入参数`chars`，则去除字符串两端**所有属于`chars`的字符**，直到遇到不属于`chars`的字符为止。
- **示例**：
```python
text = "xxxyHello, World!yyyx"
print(text.strip("xy"))  # 输出："Hello, World!"
```
3. 相关方法

|方法|作用|示例|
|---|---|---|
|`x.lstrip([chars])`|仅去除左侧（开头）的字符|`" text ".lstrip()` -> `"text "`|
|`x.rstrip([chars])`|仅去除右侧（结尾）的字符|`" text ".rstrip()` -> `" text"`|
***
### 解释 `mask = (t_data >= t_start) & (t_data <= t_end)` 语法
这是一个使用 NumPy 数组的条件筛选语句，创建了一个布尔掩码（boolean mask），用于选择满足特定条件的数据点。这种布尔掩码索引是 NumPy 最强大和最高效的特性之一，特别适合处理大型数据集，比使用循环筛选快几个数量级。

**分解说明**
1. `t_data >= t_start`：创建一个布尔数组，其中每个元素表示 `t_data` 中对应位置的值是否大于等于 `t_start`
2. `t_data <= t_end`：创建一个布尔数组，其中每个元素表示 `t_data` 中对应位置的值是否小于等于 `t_end`
3. `&`：按位与运算符，组合两个布尔数组（要求两个条件同时满足）
4. 结果是一个布尔数组（掩码），其中 `True` 表示满足条件 `t_start <= t <= t_end` 的数据点

**示例代码**
```python
import numpy as np

# 创建示例时间数据（10个点）
t_data = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# 设置筛选区间
t_start = 20
t_end = 60

# 创建布尔掩码
mask = (t_data >= t_start) & (t_data <= t_end)

print("原始时间数据:", t_data)
print("布尔掩码:   ", mask)
print("筛选结果:   ", t_data[mask])
```
**输出结果**

原始时间数据: [ 0 10 20 30 40 50 60 70 80 90]

布尔掩码:    [False False  True  True  True  True  True False False False]

筛选结果:    [20 30 40 50 60]
***
