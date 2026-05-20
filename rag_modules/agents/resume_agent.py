from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def handle(query: str) -> str:
    # query 可能是文件路径 "file:resume.pdf"
    if query.startswith("file:"):
        path = query[5:]
        loader = UnstructuredLoader(path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        chunks = text_splitter.split_documents(docs)
        # 解析出姓名、技能等...
        return f"解析到 {len(chunks)} 个文本块"
    else:
        return "未提供文件"