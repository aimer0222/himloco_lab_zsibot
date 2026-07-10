#!/usr/bin/env python3
"""
生成悬浮楼梯 MuJoCo 场景 XML
可预览四个关键变量：楼梯高度、楼梯深度、楼梯长度、楼梯厚度，以及顶部大平台
"""

# ============================================================
#  请在这里修改参数，然后运行: python gen_floating_stairs.py
# ============================================================

NUM_STAIRS    = 10     # 楼梯级数
STAIR_HEIGHT  = 0.18  # 单级台阶高度 (m)
STAIR_DEPTH   = 0.35  # 单级台阶深度 (m)
STAIR_LENGTH  = 1.2   # 单级台阶长度/宽度 (m)，即 Y 轴方向
STAIR_THICK  = 0.04  # 单级台阶厚度 (m)，即台阶板的厚度

PLATFORM_LENGTH = 3.0  # 顶部平台 Y 方向长度
PLATFORM_DEPTH = 2.0  # 顶部平台 X 方向（前后）长度
PLATFORM_THICK = 0.08 # 平台厚度

STAIR_SPACING_X = -0.05  # 台阶前后间距（留空隙方便落脚）
ROBOT_START_X   = 0.3  # 机器人起始 X（楼梯前平地）
ROBOT_START_Y   = 0.0   # 机器人起始 Y
ROBOT_START_Z   = 0.35  # 机器人起始 Z（站立高度）

SCENE_SIZE = 25  # 地面平面大小
SCENE_EXTENT = 3.0  # 相机范围

# ============================================================
#  以下代码一般不需要修改
# ============================================================

def generate_xml():
    # 计算顶部平台位置（第一个台阶顶面）
    # 第一个台阶顶面高度
    step_top_z = STAIR_HEIGHT * NUM_STAIRS
    step_top_x = STAIR_DEPTH / 2  # 第一个台阶中心

    # 平台：顶面 = step_top_z，底面 = step_top_z - PLATFORM_THICK
    platform_z = step_top_z - PLATFORM_THICK / 2
    platform_x = step_top_x + STAIR_SPACING_X * (NUM_STAIRS - 1) + STAIR_DEPTH / 2 + PLATFORM_DEPTH / 2

    # 地面高度（用于参考可视化）
    ground_z = -0.051

    lines = []
    lines.append('<mujoco model="zsibot floating stairs">\n')
    lines.append('\n')
    lines.append('  <statistic center="0 0 {:.3f}" extent="{:.1f}" />\n'.format(
        step_top_z / 2, SCENE_EXTENT))
    lines.append('\n')
    lines.append('  <visual>\n')
    lines.append('    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />\n')
    lines.append('    <rgba haze="0.15 0.25 0.35 1" />\n')
    lines.append('    <global azimuth="-130" elevation="-20" />\n')
    lines.append('  </visual>\n')
    lines.append('\n')
    lines.append('  <asset>\n')
    lines.append('    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />\n')
    lines.append('    <texture type="2d" name="groundplane" builtin="checker" mark="edge"\n')
    lines.append('      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"\n')
    lines.append('      markrgb="0.8 0.8 0.8" width="300" height="300" />\n')
    lines.append('    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="3 3" reflectance="0.2" />\n')

    # 楼梯台阶材质（蓝色系）
    lines.append('    <material name="stair_mat" rgba="0.3 0.45 0.65 1" specular="0.3" shininess="0.3" />\n')
    # 平台材质（绿色系）
    lines.append('    <material name="platform_mat" rgba="0.25 0.55 0.3 1" specular="0.3" shininess="0.3" />\n')
    lines.append('  </asset>\n')
    lines.append('\n')
    lines.append('  <worldbody>\n')
    lines.append('    <light pos="0 0 3" dir="0 0 -1" directional="true" />\n')

    # 地面
    lines.append('    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />\n')
    lines.append('    <geom name="floor_shadow" size="{:.1f} {:.1f} 0.05" type="box" pos="0 0 {:.4f}" material="groundplane" />\n'.format(
        SCENE_SIZE, SCENE_SIZE, ground_z))

    # 楼梯前的参考辅助地面（让楼梯看起来是悬浮的，底部有支撑柱）
    # 不加这个楼梯就是纯悬浮的，不接地（更有视觉冲击力）
    # 如果你想要楼梯"悬空"，直接跳过辅助柱

    # 逐级生成台阶
    for i in range(NUM_STAIRS):
        # 台阶顶面高度（步行面高度）
        top_z = STAIR_HEIGHT * (i + 1)
        # 台阶底面高度 = 顶面 - 板厚
        bot_z = top_z - STAIR_THICK
        # 台阶中心 Z（box pos）
        center_z = (top_z + bot_z) / 2
        # 台阶中心 X：每级往前移 STAIR_SPACING_X
        center_x = STAIR_DEPTH / 2 + i * (STAIR_DEPTH + STAIR_SPACING_X)
        # 尺寸
        sx = STAIR_DEPTH
        sy = STAIR_LENGTH
        sz = STAIR_THICK   # box 厚度 = STAIR_THICK

        lines.append('    <!-- Step {} (top at z={:.3f}m, thick={:.3f}m) -->\n'.format(i + 1, top_z, STAIR_THICK))
        lines.append('    <geom pos="{:.4f} 0 {:.4f}" type="box" size="{:.4f} {:.4f} {:.4f}" material="stair_mat" />\n'.format(
            center_x, center_z, sx / 2, sy / 2, sz / 2))

    # 顶部大平台
    plat_x = step_top_x + (NUM_STAIRS - 1) * (STAIR_DEPTH + STAIR_SPACING_X) + STAIR_DEPTH / 2 + PLATFORM_DEPTH / 2
    plat_z = step_top_z - PLATFORM_THICK / 2
    lines.append('\n')
    lines.append('    <!-- Top platform (top at z={:.3f}m) -->\n'.format(step_top_z))
    lines.append('    <geom pos="{:.4f} 0 {:.4f}" type="box" size="{:.4f} {:.4f} {:.4f}" material="platform_mat" />\n'.format(
        plat_x, plat_z, PLATFORM_DEPTH / 2, PLATFORM_LENGTH / 2, PLATFORM_THICK / 2))

    lines.append('  </worldbody>\n')
    lines.append('  <include file="zsl1_sim2sim.xml" />\n')
    lines.append('</mujoco>\n')

    return ''.join(lines)


def print_preview():
    step_top_z = STAIR_HEIGHT * NUM_STAIRS
    step_top_x = STAIR_DEPTH / 2
    plat_x = step_top_x + (NUM_STAIRS - 1) * (STAIR_DEPTH + STAIR_SPACING_X) + STAIR_DEPTH / 2 + PLATFORM_DEPTH / 2
    plat_z = step_top_z - PLATFORM_THICK / 2
    total_x = plat_x + PLATFORM_DEPTH / 2  # 总 X 范围

    print("=" * 55)
    print("   悬浮楼梯场景预览")
    print("=" * 55)
    print(f"  参数设置:")
    print(f"    楼梯级数      : {NUM_STAIRS} 级")
    print(f"    单级高度      : {STAIR_HEIGHT:.3f} m")
    print(f"    单级深度      : {STAIR_DEPTH:.3f} m")
    print(f"    单级长度      : {STAIR_LENGTH:.3f} m")
    print(f"    单级厚度      : {STAIR_THICK:.3f} m")
    print(f"    台阶前后间距  : {STAIR_SPACING_X:.3f} m")
    print(f"    顶部平台深度  : {PLATFORM_DEPTH:.3f} m")
    print(f"    顶部平台长度  : {PLATFORM_LENGTH:.3f} m")
    print(f"    顶部平台厚度  : {PLATFORM_THICK:.3f} m")
    print()
    print(f"  派生结果:")
    print(f"    楼梯总高度    : {step_top_z:.3f} m")
    print(f"    楼梯总长度(X) : {total_x:.3f} m")
    print(f"    顶部平台顶面  : z = {step_top_z:.3f} m")
    print(f"    机器人起始位置: ({ROBOT_START_X:.2f}, {ROBOT_START_Y:.2f}, {ROBOT_START_Z:.2f})")
    print()
    print("  每级台阶顶面高度:")
    for i in range(NUM_STAIRS):
        print(f"    Step {i+1}: z = {STAIR_HEIGHT * (i+1):.3f} m")
    print("=" * 55)

    print()
    print("ASCII 示意图 (Y 轴俯视，X 为前进方向):")
    print()

    # 绘制简单示意图
    max_x = int(total_x * 2) + 1
    height_levels = [0.0] * max_x

    for i in range(NUM_STAIRS):
        sx = int((step_top_x + i * (STAIR_DEPTH + STAIR_SPACING_X)) * 2)
        ex = int((step_top_x + i * (STAIR_DEPTH + STAIR_SPACING_X) + STAIR_DEPTH) * 2)
        for j in range(sx, ex):
            if j < len(height_levels):
                height_levels[j] = STAIR_HEIGHT * (i + 1)

    # 平台
    plat_sx = int((plat_x - PLATFORM_DEPTH / 2) * 2)
    plat_ex = int((plat_x + PLATFORM_DEPTH / 2) * 2)
    for j in range(max(0, plat_sx), min(len(height_levels), plat_ex)):
        height_levels[j] = step_top_z

    # 打印高度条
    for level in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        bar = ""
        for h in height_levels[:min(max_x, 40)]:
            bar += "█" if h >= level else " "
        print(f"  {level:.2f}m |{bar}|")

    base_bar = ""
    for _ in range(min(max_x, 40)):
        base_bar += "▁"
    print(f"  0.00m +{base_bar}+")
    print(f"         {'0':^6}{'1':^6}{'2':^6}{'3':^6}{'4':^6}m")
    print(f"         X 方向 (前进方向 ↑)")
    print()


if __name__ == "__main__":
    print_preview()
    xml = generate_xml()

    output_path = "zsibot_floating_stairs_gen.xml"
    with open(output_path, "w") as f:
        f.write(xml)
    print(f"场景文件已生成: {output_path}")
    print()
    print("用以下命令预览（需要 MuJoCo Python）:")
    print(f"  python -c \"import mujoco; m=mujoco.MjModel.from_xml_path('{output_path}'); d=mujoco.MjData(m); mujoco.viewer.launch_passive(m,d)\"")
