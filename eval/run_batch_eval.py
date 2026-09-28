import os
import subprocess
import sys
import re

def main():
    # 定义你要运行的5组方法组合
    method_groups = [
        ["DQN", "PPO", "A2C", "FinGPT", "FinMem(old)", "Full"],
        ["FinMem(old)", "Full"],
        ["FinMem(old)", "Full(CFR)"],
        ["FinMem(old)", "Full(ETR)"],
        ["FinMem(old)", "Full(RGR)"]
    ]

    # 结果保存的根目录
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

    # 运行的评估脚本列表
    target_scripts = [
        "eval/visualize_aapl.py",
        "eval/metrics_aapl.py",
        "eval/regime_analysis.py",
        "eval/wilcoxon_aapl.py"
    ]

    for methods in method_groups:
        # 将方法名拼接作为文件夹名，并替换掉括号等特殊字符以免路径报错
        raw_group_name = "_vs_".join(methods)
        group_name = re.sub(r'[<>()|?*:/\\"]', '', raw_group_name)

        output_dir = os.path.join(base_results_dir, group_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"========== 开始处理分组: {group_name} ==========")

        # 依次运行四个评估脚本
        for script in target_scripts:
            cmd = [
                sys.executable, script,
                "--methods"
            ] + methods + [
                "--output_dir", output_dir
            ]

            print(f"[{group_name}] 正在执行: {script}")
            print(f"执行命令: {' '.join(cmd)}")

            try:
                subprocess.run(cmd, check=True)
                print(f"✓ {script} 运行完成\n")
            except subprocess.CalledProcessError as e:
                print(f"✗ {script} 运行失败! 错误信息: {e}\n")

        print(f"========== {group_name} 所有脚本执行完毕 ==========\n")

if __name__ == "__main__":
    main()
