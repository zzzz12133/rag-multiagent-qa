# -*- coding: utf-8 -*-
"""文档入库脚本

用法:
    python scripts/ingest_docs.py --dir data/sample_docs
    python scripts/ingest_docs.py --file path/to/doc.pdf
"""
import argparse
import logging

from app.rag import loader, splitter, vectorstore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="企业知识库文档入库")
    parser.add_argument("--dir", help="文档目录")
    parser.add_argument("--file", help="单个文件")
    parser.add_argument("--no-recursive", action="store_true", help="目录不递归")
    args = parser.parse_args()

    path = args.dir or args.file
    if not path:
        parser.error("必须提供 --dir 或 --file")

    docs = loader.load_documents(path, recursive=not args.no_recursive)
    print(f"解析文档 {len(docs)} 篇")

    chunks = splitter.split_documents(docs)
    print(f"切分得到 {len(chunks)} 个片段")

    vectorstore.build_vector_store(chunks)
    vectorstore.build_bm25(chunks)
    print("入库完成 ✔")


if __name__ == "__main__":
    main()
