import os
from dotenv import load_dotenv
from controller.chat_controller import ChatController
from view.cli_view import CLIView

def main():
    load_dotenv()
    # api_key = os.getenv("ZHIPU_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请在 .env 中配置 ZHIPU_API_KEY")
    
    controller = ChatController()
    view = CLIView()
    view.run(controller)

if __name__ == "__main__":
    main()