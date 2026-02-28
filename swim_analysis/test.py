from graphviz import Digraph


def create_architecture_diagram():
    # 创建有向图
    dot = Digraph('Swim_AI_Architecture', comment='游泳姿态分析系统架构')
    dot.attr(rankdir='TB', size='10,10')  # 从上到下布局
    dot.attr('node', shape='box', style='filled', fontname='Microsoft YaHei')

    # 1. 输入层
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='输入层 (Input Layer)', color='lightgrey', style='dashed')
        c.node('Video', '输入视频流\n(Video Input)', fillcolor='azure')
        c.node('Enhance', '图像增强模块\n(CLAHE / LAB色彩空间)', fillcolor='azure')

    # 2. 核心模型层
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='AI推理核心 (Core Inference)', color='lightgrey', style='dashed')
        c.node('YOLO', 'YOLOv8-Pose 模型\n(人体关键点检测)', shape='ellipse', fillcolor='gold')
        c.node('Keypoints', '17个骨骼关键点数据\n(Keypoints Data)', fillcolor='lightyellow')

    # 3. 业务逻辑层 (三大分支)
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='功能逻辑层 (Functional Logic)', color='grey')

        # 分支 1: 视觉反馈
        c.node('AngleCalc', '肢体角度计算\n(肩-肘-腕)', fillcolor='lavender')
        c.node('RealTimeUI', '实时错误判定\n(高肘/直臂检测)', fillcolor='lavender')

        # 分支 2: 划频分析
        c.node('SignalProc', '信号处理\n(数据平滑/汉宁窗)', fillcolor='mistyrose')
        c.node('DataRepair', '数据修复\n(线性插值补全)', fillcolor='mistyrose')
        c.node('PeakDetect', '波峰检测 & SPM计算\n(Stroke Rate Analysis)', fillcolor='mistyrose')

        # 分支 3: 动作对比
        c.node('SeqExtract', '动作序列提取', fillcolor='honeydew')
        c.node('DTW', 'DTW 动态时间规整\n(序列对齐 & 评分)', fillcolor='honeydew')

    # 4. 输出层
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='输出层 (Output Layer)', color='lightgrey', style='dashed')
        c.node('Out_Video', '带标注的反馈视频', shape='note')
        c.node('Out_Chart', '划频分析折线图', shape='note')
        c.node('Out_Score', '动作评分与对比图', shape='note')

    # === 定义连接关系 ===

    # 输入流
    dot.edge('Video', 'Enhance', label='帧预处理')
    dot.edge('Enhance', 'YOLO', label='增强帧')
    dot.edge('YOLO', 'Keypoints', label='输出 (x,y,conf)')

    # 分发到三大功能模块
    dot.edge('Keypoints', 'AngleCalc', label='模式1: 实时反馈')
    dot.edge('Keypoints', 'DataRepair', label='模式2: 划频分析')
    dot.edge('Keypoints', 'SeqExtract', label='模式3: 动作对比')

    # 逻辑流 1 (视觉反馈)
    dot.edge('AngleCalc', 'RealTimeUI')
    dot.edge('RealTimeUI', 'Out_Video')

    # 逻辑流 2 (划频分析)
    dot.edge('DataRepair', 'SignalProc', label='修复断点')
    dot.edge('SignalProc', 'PeakDetect', label='降噪')
    dot.edge('PeakDetect', 'Out_Chart')

    # 逻辑流 3 (动作对比)
    dot.edge('SeqExtract', 'DTW', label='用户序列 vs 专业序列')
    dot.edge('DTW', 'Out_Score')

    # 保存并渲染
    # 生成 pdf 和 png
    dot.render('swim_ai_architecture', format='png', view=False)
    print("架构图已生成: swim_ai_architecture.png")


if __name__ == '__main__':
    create_architecture_diagram()