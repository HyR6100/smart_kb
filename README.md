1. 知识提取接口 (Extract)

将非结构化文本转换为 Neo4j 节点和关系。

终端命令：
Bash

curl -X POST http://127.0.0.1:8000/api/extract/ \
     -H "Content-Type: application/json" \
     -d '{
       "text": "EmoBot是一款运行在Ubuntu 22.04上的多模态机器人，它整合了Qwen 2.5作为决策大脑。"
     }'

    功能描述：调用 LLM 识别文本中的实体（如 EmoBot, Ubuntu, Qwen 2.5）及其关系，并自动同步到 Neo4j 数据库及内存缓存中。

    预期反馈：返回包含已提取三元组信息的 JSON 数据。

2. 知识问答接口 (Ask/GraphRAG)

基于已有的知识图谱进行精准检索问答。

终端命令：
Bash

curl -X POST http://127.0.0.1:8000/api/ask/ \
     -H "Content-Type: application/json" \
     -d '{
       "question": "EmoBot运行在什么操作系统上？它的决策大脑是什么？"
     }'

    功能描述：系统首先从问题中识别实体，在 Neo4j 中检索相关子图作为背景上下文，最后由 LLM 生成准确答案。

    预期反馈：
    JSON

    {
      "answer": "EmoBot运行在Ubuntu 22.04操作系统上，其决策大脑整合了Qwen 2.5模型。"
    }
