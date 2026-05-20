class CLIView:
    def run(self, controller):
        print("招聘AI助手已启动（输入 'quit' 退出）")
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() == "quit":
                break
            response = controller.process_message(user_input)
            print(response)