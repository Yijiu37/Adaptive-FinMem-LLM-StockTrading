"""
把 test 阶段结束后的 agent 动作导出为 CSV。

读取: data/09_results/agent_1   (run_opeai.sh 的 -rp 指向这里)
写出: aapl_gpt4.csv              (eval/ 脚本默认读这个文件)
"""
from puppy import LLMAgent

if __name__ == "__main__":
    agent = LLMAgent.load_checkpoint("./data/09_results/agent_1")
    df = agent.portfolio.get_action_df()
    df.write_csv("aapl_gpt4.csv")
    print(f"✓ 已导出 {df.shape[0]} 条动作 → aapl_gpt4.csv")
    print(df.tail(5))