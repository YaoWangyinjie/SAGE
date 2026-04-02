"""
SAGE — 命令行入口。

用法：
  python main.py "分析 EAGLE 的部署挑战"
  python main.py --interactive
  python main.py --demo
  python main.py --add-path examples/eagle_path.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAGE — Speculative Agent with Graph-based Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py "分析 EAGLE 的部署挑战"
  python main.py --interactive
  python main.py --demo
  python main.py --add-path examples/paths/eagle_path.json
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="要处理的查询（单次模式）",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="进入交互式对话模式",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行内置演示查询",
    )
    parser.add_argument(
        "--add-path",
        metavar="JSON_FILE",
        help="向推理图数据库中添加路径模板（JSON 文件路径）",
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="展示当前推理图中的所有路径",
    )
    return parser.parse_args()


async def run_single_query(agent, query: str):
    """执行单次查询并打印结果"""
    print(f"\n🔍 查询：{query}")
    print("=" * 60)
    answer = await agent.query(query)
    print(f"\n📋 答案：\n{answer}")
    print("=" * 60)


async def run_interactive(agent):
    """交互式模式"""
    print("\n=== SAGE 交互模式 ===")
    print("输入查询后按回车，输入 'quit' 或 'exit' 退出\n")
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        await run_single_query(agent, query)


async def run_demo(agent):
    """内置演示：运行一组典型查询"""
    demo_queries = [
        "分析 EAGLE 论文的核心技术原理和部署挑战",
        "GPT-4 和 Claude 3 在推理能力上有什么区别？",
        "介绍一下 Speculative Decoding 的基本原理",
    ]
    print("\n=== SAGE 演示模式 ===\n")
    for q in demo_queries:
        await run_single_query(agent, q)
        print()


def add_path_from_file(agent, json_file: str):
    """从 JSON 文件加载推理路径模板并添加到图数据库"""
    from src.graph.reasoning_graph import ReasoningGraphDB
    from src.data_structures import ReasoningPath, GraphNode, GraphEdge

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = ReasoningGraphDB._deserialize_path(data)
        agent.graph_db.add_path(path)
        agent.graph_db.save()
        print(f"✅ 成功添加路径: {path.path_id}（共 {len(path.nodes)} 个 hop）")
    except FileNotFoundError:
        print(f"❌ 文件不存在: {json_file}")
    except Exception as e:
        print(f"❌ 加载路径失败: {e}")


def show_graph(agent):
    """展示推理图中所有路径"""
    paths = agent.graph_db._paths
    if not paths:
        print("推理图为空（尚未学习任何路径）")
        return
    print(f"\n=== 推理图（共 {len(paths)} 条路径）===")
    for pid, path in paths.items():
        print(f"\n[{pid}] {path.query_template}")
        print(f"  实体类型: {path.entity_types}")
        print(f"  hop 数: {len(path.nodes)}，使用次数: {path.use_count}")
        for node in path.nodes:
            print(f"    - {node.node_id}: {node.intent_template} ({node.tool_name})")


async def main():
    args = parse_args()

    # 懒加载 Agent（避免解析参数时就初始化模型）
    from src.agent import SAGEAgent
    agent = SAGEAgent()

    try:
        if args.add_path:
            add_path_from_file(agent, args.add_path)

        elif args.show_graph:
            show_graph(agent)

        elif args.demo:
            await run_demo(agent)

        elif args.interactive:
            await run_interactive(agent)

        elif args.query:
            await run_single_query(agent, args.query)

        else:
            # 无参数：进入交互模式
            await run_interactive(agent)

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
