import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量存储数据和拟合参数
# t_data: 存储从用户输入解析得到的时间数据，类型为可选的NumPy数组
#          - 初始值为None，表示尚未加载数据
#          - 解析成功后存储时间数据点的一维数组
t_data: Optional[np.ndarray] = None

# J_data: 存储从用户输入解析得到的通量数据，类型为可选的NumPy数组
#          - 初始值为None，表示尚未加载数据
#          - 解析成功后存储通量数据点的一维数组
#          - 与t_data长度相同，形成一一对应的数据点对
J_data: Optional[np.ndarray] = None

# current_fig: 存储当前显示的绘图对象
#               - 用于管理绘图区域，当更新图表时销毁旧图表并创建新图表
#               - 初始值为None，表示尚未创建图表
current_fig = None

# fit_params: 存储模型拟合结果的字典
#             - 'k': 拟合得到的污染系数k值
#             - 'J_star': 拟合得到的平衡通量J*值
#             - 'r_squared': 拟合的R²决定系数，衡量拟合优度
#             - 所有键初始值为None，表示尚未进行拟合
fit_params = {'k': None, 'J_star': None, 'r_squared': None}


def parse_column(txt):
    """解析单列数据
    将用户输入的文本数据（逗号或换行分隔）转换为浮点数列表

    参数:
        txt (str): 包含数值数据的文本字符串，数值间用逗号或换行分隔

    返回值:
        list[float] | None: 成功时返回浮点数列表，失败时弹出错误提示并返回None
    """
    try:
        # 统一分隔符：将换行符替换为逗号，使所有数据在同一分隔符体系下
        unified = txt.replace('\n', ',')

        # 按逗号分割字符串：生成原始数据字符串列表
        # 例如 "1,2,3" -> ['1', '2', '3']
        raw = unified.split(',')

        # 数据清洗和转换:
        #   a. 遍历分割后的每个元素
        #   b. 跳过空字符串（strip()后长度为0的元素）
        #   c. 去除首尾空格后转换为浮点数
        #   例如 [' 1 ', '  ', '2.5'] -> [1.0, 2.5]
        return [float(x.strip()) for x in raw if x.strip()]

    except Exception as e:
        # 异常处理：捕获任何转换失败的情况（如非数字字符）
        messagebox.showerror("数据错误", f"数据解析失败：{str(e)}")
        return None


def parse_data():
    """解析两列数据
    从用户界面中的两个文本框中提取时间(t)和通量(J)数据，进行验证和存储

    处理流程:
    1. 从时间(t)文本框和通量(J)文本框中获取原始文本数据
    2. 分别调用parse_column函数解析单列数据
    3. 检查两列数据是否都成功解析
    4. 验证两列数据长度是否一致
    5. 将解析后的数据转换为NumPy数组并存储到全局变量

    返回值:
        bool: 如果数据成功解析并存储返回True，否则返回False
    """
    # 声明使用全局变量，用于存储解析后的时间(t)和通量(J)数据
    global t_data, J_data

    # 从时间(t)文本框中获取内容
    # "1.0" 表示从第一行第一列开始
    # "end-1c" 表示到文本末尾的前一个字符（排除自动添加的换行符）
    t_values = parse_column(t_text.get("1.0", "end-1c"))

    # 从通量(J)文本框中获取内容
    j_values = parse_column(J_text.get("1.0", "end-1c"))

    # 检查两列数据是否都成功解析（非None）
    if t_values and j_values:
        # 验证两列数据长度是否一致
        if len(t_values) != len(j_values):
            # 长度不一致时显示错误提示
            messagebox.showerror("数据错误", "t和J数据长度不一致")
            return False

        # 将解析后的列表转换为NumPy数组
        t_data = np.array(t_values)
        J_data = np.array(j_values)

        # 返回成功标志
        return True

    # 如果任一列数据解析失败，返回False
    return False


def fit_model():
    """执行膜污染模型的拟合计算
    主要流程：
    1. 解析用户输入的时间(t)和通量(J)数据
    2. 获取用户选择的模型类型和拟合区间
    3. 准备微分方程和拟合模型
    4. 使用最小二乘法进行曲线拟合
    5. 计算拟合优度(R²)
    6. 更新结果显示并绘制拟合曲线图
    """
    # 声明使用全局变量：current_fig（当前图表）和 fit_params（拟合参数）
    global current_fig, fit_params

    # 步骤1: 数据解析
    # 检查数据是否有效（调用parse_data()并确保t_data不为空）
    if not parse_data() or t_data is None:
        return

    try:
        # 步骤2: 获取用户输入参数
        # 从下拉菜单获取选择的模型指数n
        n = model_var.get()
        # 从输入框获取拟合起始时间和结束时间
        t_start = float(start_entry.get())
        t_end = float(end_entry.get())
    except ValueError:
        # 输入值转换失败时显示错误
        messagebox.showerror("错误", "参数输入不合法")
        return

    # 模型指数映射关系：将界面选择的n值转换为微分方程中的指数m
    # n=2 (完全阻塞) → m=0
    # n=1.5 (标准阻塞) → m=0.5
    # n=1 (中间阻塞) → m=1
    # n=0.5 (滤饼层) → m=2
    m_dict = {'2': 0, '1.5': 0.5, '1': 1, '0.5': 2}
    m = m_dict[n]  # 获取对应的指数值

    # 创建数据选择掩码：仅选择在拟合区间[t_start, t_end]内的数据点
    mask = (t_data >= t_start) & (t_data <= t_end)
    # 检查区间内是否有足够的数据点（至少2个点才能拟合）
    if sum(mask) < 2:
        messagebox.showerror("错误", "拟合区间数据不足")
        return

    # 提取拟合区间内的数据
    t_fit = t_data[mask]
    j_fit = J_data[mask]

    # 步骤3: 定义微分方程
    def differential_eq(_, j, k, j_star):
        """膜污染动力学微分方程
        方程形式: dJ/dt = -k * J^m * (J - J*)
        参数:
            _ : 时间t（方程不显含t，但为满足scipy接口要求保留）
            j : 当前通量值
            k : 污染系数（待拟合参数）
            j_star : 平衡通量（待拟合参数）
        返回值:
            当前时刻的通量变化率
        """
        return -k * (j ** m) * (j - j_star)

    # 定义拟合模型函数
    def model(_, k, j_star):
        """数值求解微分方程生成拟合曲线
        参数:
            _ : 时间点数组（由curve_fit自动传入）
            k : 污染系数
            j_star : 平衡通量
        返回值:
            模型预测的通量值数组
        """
        # 使用solve_ivp求解微分方程
        sol = solve_ivp(
            differential_eq,              # 微分方程
            [t_start, t_end],       # 积分时间范围
            [j_fit[0]],               # 初始条件（使用第一个数据点的通量值）
            args=(k, j_star),             # 传递给微分方程的参数
            t_eval=t_fit,                 # 指定输出的时间点（与输入数据时间点对应）
            dense_output=True             # 生成连续解（提高精度）
        )
        return sol.y[0]  # type: ignore   # 返回通量解（忽略类型检查警告）

    # 初始化默认参数
    k_fit = 0.01  # 污染系数k的初始值
    j_star_fit = 1.0  # 平衡通量J*的初始值
    success = True  # 拟合成功标志（默认为True）

    # 步骤4: 执行曲线拟合
    try:
        # 使用curve_fit进行非线性最小二乘拟合
        result = curve_fit(
            model,  # 要拟合的模型函数
            t_fit,  # 自变量数据（时间）
            j_fit,  # 因变量数据（实际通量值）
            p0=[k_fit, j_star_fit],  # 参数初始猜测值
            maxfev=1000  # 最大函数评估次数
        )
        popt = result[0]  # 提取最优参数
        k_fit, j_star_fit = popt  # 解包参数

    except RuntimeError as e:
        # 处理拟合失败的情况
        if 'Optimal parameters not found' in str(e):
            # 未找到最优参数（收敛失败）但仍使用当前最佳结果
            success = False
            messagebox.showwarning("警告", "拟合未完全收敛，显示当前最佳结果")
        else:
            # 其他运行时错误
            messagebox.showerror("拟合错误", str(e))
            return

    try:
        # 步骤5: 计算拟合结果和统计量

        # 使用拟合得到的参数计算预测值
        j_pred = model(t_fit, k_fit, j_star_fit)

        # 计算R²决定系数（拟合优度）
        ss_res = np.sum((j_fit - j_pred) ** 2)  # 残差平方和（预测值与实际值之差的平方和）
        ss_tot = np.sum((j_fit - np.mean(j_fit)) ** 2)  # 总平方和（实际值与其均值之差的平方和）
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0  # 计算R²（避免分母为零的情况）

        # 更新全局拟合参数
        fit_params.update({'k': k_fit, 'J_star': j_star_fit, 'r_squared': r_squared})
        # 在界面上显示拟合结果
        result_text.set(f"K: {k_fit:.4f}\nJ*: {j_star_fit:.4f}\nR²: {r_squared:.4f}")

        # 步骤6: 绘制拟合结果图

        # 创建新图形
        fig = plt.figure(figsize=(6, 4))
        # 绘制原始数据点（蓝色圆点）
        plt.plot(t_fit, j_fit, 'bo', label='原始数据')
        # 绘制拟合曲线
        plt.plot(t_fit, j_pred,
                 'r-' if success else 'r--',  # 成功：红色实线，未收敛：红色虚线
                 label=f'拟合曲线 (n={n}){"*" if not success else ""}')  # 未收敛时添加星号标记
        # 设置坐标轴范围（x轴扩展10%，y轴扩展10%）
        plt.xlim(t_start - 0.1 * (t_end - t_start), t_end + 0.1 * (t_end - t_start))
        plt.ylim(min(j_fit) - 0.1, max(j_fit) + 0.1)
        # 添加拟合区间边界线（绿色虚线表示开始，品红虚线表示结束）
        plt.axvline(t_start, color='g', ls='--', label='开始拟合位置')
        plt.axvline(t_end, color='m', ls='--', label='结束拟合位置')
        # 添加图例（右上角）
        plt.legend(loc='upper right')
        # 设置坐标轴标签
        plt.xlabel('时间 t')
        plt.ylabel('通量 J')

        # 更新GUI中的图表显示
        if current_fig:
            # 销毁旧图表（如果存在）
            current_fig.get_tk_widget().destroy()
        # 创建新的图表组件
        current_fig = FigureCanvasTkAgg(fig, master=plot_frame)
        # 渲染图表
        current_fig.draw()
        # 将图表组件放入界面
        current_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    except Exception as e:
        # 处理绘图过程中的异常
        messagebox.showerror("绘图错误", str(e))


def save_results():
    """保存拟合结果到CSV文件

        功能说明:
        1. 检查是否已执行拟合（是否有有效的拟合参数）
        2. 弹出文件保存对话框让用户选择保存位置
        3. 重新计算拟合曲线（使用拟合得到的参数）
        4. 将原始数据、预测值和拟合参数保存到CSV文件
        5. 显示保存成功或失败的消息

        文件格式:
            t, J_actual, J_predicted
            [时间1], [实际通量1], [预测通量1]
            [时间2], [实际通量2], [预测通量2]
            ...
    """
    # 1. 检查是否已执行拟合
    # 通过检查 fit_params['k'] 是否存在来判断
    if not fit_params['k']:
        # 如果尚未拟合，显示错误提示
        messagebox.showerror("错误", "请先执行拟合")
        return  # 终止函数执行

    # 2. 弹出文件保存对话框
    # defaultextension=".csv" 确保默认保存为CSV格式
    file_path = filedialog.asksaveasfilename(defaultextension=".csv")

    # 如果用户选择了文件路径（不是取消对话框）
    if file_path:
        try:
            # 3. 重新计算拟合曲线
            # 获取当前选择的模型类型
            n = model_var.get()
            # 模型指数映射（与拟合时相同）
            m = {'2': 0, '1.5': 0.5, '1': 1, '0.5': 2}[n]

            # 获取拟合区间参数
            t_start = float(start_entry.get())
            t_end = float(end_entry.get())

            # 创建数据掩码，选择拟合区间内的数据点
            mask = (t_data >= t_start) & (t_data <= t_end)
            # 提取时间数据
            t_fit = t_data[mask]

            # 定义微分方程（与拟合时相同）
            def differential_eq(_, j, k, j_star):
                """膜污染动力学微分方程"""
                return -k * (j ** m) * (j - j_star)

            # 使用拟合得到的参数求解微分方程
            # 初始条件使用区间内第一个数据点的通量值
            sol = solve_ivp(
                differential_eq,  # 微分方程
                [t_start, t_end],  # 时间范围
                [J_data[mask][0]],  # 初始条件（第一个数据点的通量值）
                args=(fit_params['k'], fit_params['J_star']),  # 拟合参数
                t_eval=t_fit  # 指定输出的时间点
            )

            # 4. 准备保存数据
            # 将时间、实际通量、预测通量合并为二维数组
            # np.column_stack() 按列堆叠数组
            save_data = np.column_stack((t_fit, J_data[mask], sol.y[0]))  # type: ignore

            # 5. 保存到CSV文件
            np.savetxt(
                file_path,  # 文件路径
                save_data,  # 数据数组
                delimiter=',',  # 分隔符（逗号）
                header='t,J_actual,J_predicted',  # 列标题
                comments=''  # 不在标题行前添加注释字符
            )

            # 显示保存成功消息
            messagebox.showinfo("成功", "文件保存成功！")

        except Exception as e:
            # 处理保存过程中的任何异常
            messagebox.showerror("错误", f"保存失败：{str(e)}")


# 主窗口配置
root = tk.Tk()  # 创建主窗口对象
root.title("膜污染阻塞模型拟合")  # 设置窗口标题

# 左侧控制面板
# 创建一个框架用于放置所有控制组件
control_frame = ttk.Frame(root, padding=10)  # 添加10像素的内边距
control_frame.pack(side=tk.LEFT, fill=tk.Y)  # 放置在窗口左侧，垂直方向填充可用空间

# 数据输入部分
# 创建标签框架，用于组织数据输入组件
data_frame = ttk.LabelFrame(control_frame, text="数据输入（逗号或换行分隔）")
# 使用网格布局管理器放置框架，设置垂直间距5像素，东西方向拉伸
data_frame.grid(row=0, column=0, pady=5, sticky="ew")

# 时间标签和文本框
t_label = ttk.Label(data_frame, text="时间 t：")  # 创建标签
t_label.grid(row=0, column=0, sticky="w")  # 放置在第0行第0列，靠左对齐
t_text = tk.Text(data_frame, height=6, width=25)  # 创建文本输入框，高度6行，宽度25字符
t_text.grid(row=1, column=0)  # 放置在第1行第0列

# 通量标签和文本框
J_label = ttk.Label(data_frame, text="通量 J：")  # 创建标签
J_label.grid(row=2, column=0, sticky="w", pady=(10, 0))  # 放置在第2行第0列，靠左对齐，顶部添加10像素间距
J_text = tk.Text(data_frame, height=6, width=25)  # 创建文本输入框
J_text.grid(row=3, column=0)  # 放置在第3行第0列

# 模型选择
# 创建标签框架，用于组织模型选择组件
model_frame = ttk.LabelFrame(control_frame, text="选择模型")
# 放置在第1行第0列，垂直间距10像素，东西方向拉伸
model_frame.grid(row=1, column=0, pady=10, sticky="ew")

# 创建单选按钮组
model_var = tk.StringVar(value="1.5")  # 创建字符串变量，默认值为"1.5"
# 定义模型选项列表：(显示文本, 实际值)
models = [
    ("完全阻塞 (n=2)", "2"),
    ("标准阻塞 (n=1.5)", "1.5"),
    ("中间阻塞 (n=1)", "1"),
    ("滤饼层 (n=0.5)", "0.5")
]

# 遍历模型选项并创建单选按钮
for idx, (text, value) in enumerate(models):
    # 创建单选按钮，绑定到model_var变量
    rb = ttk.Radiobutton(model_frame, text=text, value=value, variable=model_var)
    # 放置在模型框架中的第idx行第0列，靠左对齐，左侧添加5像素间距
    rb.grid(row=idx, column=0, sticky="w", padx=5)

# 拟合区间
# 创建标签框架，用于组织拟合区间组件
range_frame = ttk.LabelFrame(control_frame, text="拟合区间 (t)")
# 放置在第2行第0列，垂直间距5像素，东西方向拉伸
range_frame.grid(row=2, column=0, pady=5, sticky="ew")

# 开始时间标签和输入框
ttk.Label(range_frame, text="开始 t:").grid(row=0, column=0)  # 创建标签并放置在第0行第0列
start_entry = ttk.Entry(range_frame, width=8)  # 创建输入框，宽度8字符
start_entry.insert(0, "0")  # 设置默认值为"0"
start_entry.grid(row=0, column=1, padx=5)  # 放置在第0行第1列，左右添加5像素间距

# 结束时间标签和输入框
ttk.Label(range_frame, text="结束 t:").grid(row=1, column=0, pady=5)  # 创建标签并放置在第1行第0列，顶部添加5像素间距
end_entry = ttk.Entry(range_frame, width=8)  # 创建输入框
end_entry.insert(0, "120")  # 设置默认值为"120"
end_entry.grid(row=1, column=1, padx=5)  # 放置在第1行第1列，左右添加5像素间距

# 操作按钮
# 创建框架用于放置操作按钮
button_frame = ttk.Frame(control_frame)
button_frame.grid(row=3, column=0, pady=10)  # 放置在第3行第0列，顶部添加10像素间距

# 创建"拟合"按钮，绑定到fit_model函数
ttk.Button(button_frame, text="拟合", command=fit_model).grid(row=0, column=0, padx=5)
# 创建"保存为CSV"按钮，绑定到save_results函数
ttk.Button(button_frame, text="保存为CSV", command=save_results).grid(row=0, column=1, padx=5)

# 结果显示
# 创建标签框架，用于显示拟合结果
result_frame = ttk.LabelFrame(control_frame, text="拟合结果")
# 放置在第4行第0列，东西方向拉伸
result_frame.grid(row=4, column=0, sticky="ew")

# 创建用于显示结果的字符串变量
result_text = tk.StringVar()
# 创建标签用于显示结果文本
result_label = ttk.Label(result_frame, textvariable=result_text)
# 放置在第0行第0列，四周添加5像素间距
result_label.grid(row=0, column=0, padx=5, pady=5)

# 右侧绘图区域
# 创建框架用于放置图表
plot_frame = ttk.Frame(root)
# 放置在窗口右侧，水平和垂直方向都填充可用空间，并扩展以填充额外空间
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# 启动主事件循环
root.mainloop()
