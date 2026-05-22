import os
import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv()

    if not (os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("请在 .env 中配置 ZHIPU_API_KEY 或 OPENAI_API_KEY")

    uvicorn.run(
        "view.web_view:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # 开发阶段热重载，生产改为 False
    )


if __name__ == "__main__":
    main()
