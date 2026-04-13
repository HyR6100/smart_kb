import json
import numpy as np
import re
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# 配置neo4j的数据库连接
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

embed_model = OllamaEmbeddings(model="bge-m3")
llm = OllamaLLM(model="qwen2.5:14b")

# 内存缓存
_node_cache = {}  # {name: vector}
_rel_cache = {}   # {rel: vector}


def load_cache_from_neo4j(session):
    """首次运行时从neo4j加载已有数据到缓存"""
    if not _node_cache:
        existing_nodes = [r["name"] for r in session.run("MATCH (e:Entity) RETURN e.name AS name")]
        if existing_nodes:
            print(f"[缓存] 加载 {len(existing_nodes)} 个节点向量...")
            vecs = embed_model.embed_documents(existing_nodes)
            _node_cache.update(dict(zip(existing_nodes, vecs)))

    if not _rel_cache:
        existing_rels = list(set(
            r["rel"] for r in session.run("MATCH ()-[r]->() RETURN type(r) AS rel")
        ))
        if existing_rels:
            print(f"[缓存] 加载 {len(existing_rels)} 个关系向量...")
            vecs = embed_model.embed_documents(existing_rels)
            _rel_cache.update(dict(zip(existing_rels, vecs)))


def get_top_k_similar(query, candidates_cache, top_k=5):
    """从缓存中找出最相似的top_k个候选"""
    if not candidates_cache:
        return []
    query_vec = embed_model.embed_query(query)
    names = list(candidates_cache.keys())
    vecs = list(candidates_cache.values())
    scores = cosine_similarity([query_vec], vecs)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(names[i], scores[i]) for i in top_indices if scores[i] > 0.5]


def normalize_by_llm(new_data):
    """向量召回候选后，LLM做最终标准化（节点+关系一起处理）"""
    # 为每个新节点找候选
    node_candidates = {}
    for node in new_data.get("nodes", []):
        similar = get_top_k_similar(node, _node_cache, top_k=5)
        if similar:
            node_candidates[node] = [name for name, _ in similar]

    # 为每个新关系找候选
    rel_candidates = {}
    for edge in new_data.get("edges", []):
        if isinstance(edge, list) and len(edge) >= 3:
            rel = edge[1]
            similar = get_top_k_similar(rel, _rel_cache, top_k=5)
            if similar:
                rel_candidates[rel] = [name for name, _ in similar]

    # 没有候选说明是全新图谱，直接返回
    if not node_candidates and not rel_candidates:
        return new_data

    prompt = f"""
    你是知识图谱专家，对新数据进行标准化处理。

    新数据：{json.dumps(new_data, ensure_ascii=False)}

    节点候选映射（如果新节点与候选是同一实体，用候选名替换）：
    {json.dumps(node_candidates, ensure_ascii=False)}

    关系候选映射（如果新关系与候选语义相同，用候选名替换）：
    {json.dumps(rel_candidates, ensure_ascii=False)}

    规则：
    1. 只有确定是同一实体/语义才替换，不确定保留原名
    2. 关系名不超过4个字
    3. 只输出JSON，无任何解释

    格式：
    {{
    "nodes": ["实体A", "实体B"],
    "edges": [["实体A", "关系", "实体B"]]
    }}
    """
    response = llm.invoke(prompt)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return new_data
        return json.loads(match.group())
    except:
        return new_data


def save_to_neo4j(data):
    """将信息存入neo4j"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:

        # 加载缓存
        load_cache_from_neo4j(session)

        # 标准化（节点去重 + 关系去重一起做）
        data = normalize_by_llm(data)
        print(f"[标准化后] {json.dumps(data, ensure_ascii=False)}")

        # 存入节点
        for node_name in data.get("nodes", []):
            session.run("MERGE (e:Entity {name: $name})", name=node_name)
            if node_name not in _node_cache:
                _node_cache[node_name] = embed_model.embed_query(node_name)

        # 存入关系
        for edge in data.get('edges', []):
            if not isinstance(edge, list) or len(edge) < 3:
                continue
            source, rel_type, target = edge[0], edge[1], edge[2]
            final_rel = re.sub(r'[^\w\u4e00-\u9fff]', '_', rel_type)
            session.run(f"""
                MATCH (a:Entity {{name: $source}})
                MATCH (b:Entity {{name: $target}})
                MERGE (a)-[r:`{final_rel}`]->(b)
            """, source=source, target=target)
            print(f"[写入] '{source}'-[{final_rel}]->'{target}'")
            if rel_type not in _rel_cache:
                _rel_cache[rel_type] = embed_model.embed_query(rel_type)

    driver.close()


def extract_relations(text):
    """从文本提取实体和关系，动态Schema约束从源头减少歧义"""

    # 从缓存拿已有关系词表，约束LLM输出
    existing_rels = list(_rel_cache.keys())
    existing_nodes = list(_node_cache.keys())

    schema_hint = ""
    if existing_rels:
        schema_hint += f"\n已有关系词表（语义相近必须复用，不要造新词）：{existing_rels}"
    if existing_nodes:
        schema_hint += f"\n已有实体名称（同一实体必须用完全相同的名字）：{existing_nodes}"

    prompt = f"""
    你是一个AI Agent领域的专家。请从以下文本中提取实体和他们之间的关系，返回JSON格式。
    输出规范：
    1. 必须只输出JSON，严禁包含任何解释、前言或后缀
    2. 关系名必须是简洁动词，不超过4个字
    3. 同一语义只能用一个词，禁止出现"训练于"和"在...中训练"这种重复
    4. 实体名称不要带形容词{schema_hint}

    格式：
    {{
    "nodes": ["实体A", "实体B"],
    "edges": [["实体A", "关系", "实体B"]]
    }}
    文本：{text}
    """
    response = llm.invoke(prompt)

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            raise ValueError("未找到JSON结构")
        data = json.loads(match.group())
        save_to_neo4j(data)
        return data
    except Exception as e:
        return {"error": f"解析失败：{str(e)}", "error_response": response}
    
def extract_entities_from_query(query):
    """寻找已存在的相关实体"""
    existing_nodes = list(_node_cache.keys())
    prompt = f"""
    你是一个知识图谱检索专家，根据用户的问题整理出哪些实体已存在于已知实体列表里。
    已知实体列表：{existing_nodes}
    用户的问题：{query}
    要求: 1. 只输出实体的名字，多个实体用逗号分割 2. 如果没有，则输出[] 3. 只输出一个json, 不要解释, 不要包裹在其他key里 （必须遵守）
    示例：["咖啡", "茶多酚", "咖啡因"]
    """
    response = llm.invoke(prompt).strip()
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if not match:
            return []
        return json.loads(match.group())
    except:
        return []

def apply_neo4j_query(query):
    """使用neo4j参与用户问答"""
    # 提取已经存在的实体
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        load_cache_from_neo4j(session)
    existed_entities = extract_entities_from_query(query)
    print(f"DEBUG: 识别到的实体列表 -> {existed_entities}") # 打印识别到的实体

    if not existed_entities:
        return llm.invoke(query)
    
    # 已存在的实体查询relations
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        query_context = """
        MATCH path = (n:Entity)-[r*1..3]->(m:Entity)
        WHERE n.name IN $names OR m.name IN $names
        RETURN [node in nodes(path) | node.name] as nodes,
               [rel in relationships(path) | type(rel)] as rels
        LIMIT 20
        """
        query_results = session.run(query_context, names=existed_entities)
        # 处理查询结果
        relations = []
        for r in query_results:
            nodes = r['nodes']
            rels = r['rels']
            # 把路径拼成可读字符串，例如 (A)-[rel1]->(B)-[rel2]->(C)
            path_str = ""
            for i, rel in enumerate(rels):
                path_str += f"({nodes[i]})-[{rel}]->"
            path_str += f"({nodes[-1]})"
            relations.append(path_str)
        neo4j_information = "\n".join(relations)
        print(f"DEBUG: 检索到的子图背景 -> \n{neo4j_information}") # 打印检索到的知识
    
    # 用查询结构构造prompt
    prompt = f"""
    你是一个专业的 AI 助手。请结合以下从知识图谱中检索到的背景信息来回答用户的问题。
    如果背景信息不足以回答，请结合你的通用知识，但优先参考背景信息。

    背景信息：{neo4j_information}
    用户问题：{query}
    """
    return llm.invoke(prompt)
