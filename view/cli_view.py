"""
cli_view.py — 命令行视图

从 ChatController 接收统一响应字典，取出 message 字段展示给用户。
"""


class CLIView:
    def run(self, controller):
        print("招聘AI助手已启动（输入 'quit' 退出，输入 'stats' 查看路由统计）")
        while True:
            user_input = input("\n用户: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("再见！")
                break
            if user_input.lower() == "stats":
                print(controller.get_routing_stats())
                continue

            response = controller.process_message(user_input)

            # 统一响应字典 → 取 message 展示
            message = response.get("data", {}).get("message", "（无响应）")
            status  = response.get("status", "success")

            if status == "error":
                print(f"\n⚠️  {message}")
            else:
                print(f"\n小才: {message}")
