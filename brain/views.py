from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from .services import extract_relations, apply_neo4j_query

class ExtractTestView(APIView):
    def post(self, request):
        text = request.data.get('text', '')
        if not text:
            return Response({"error": "请输入文字"}, status=400)
        
        result = extract_relations(text)

        return Response(result)
    
class AskKnowledgeView(APIView):
    def post(self, request):
        question = request.data.get('question', '')
        if not question:
            return Response({"error": "请输入问题"}, status=400)
        
        # 调用 services.py 里的 GraphRAG 逻辑
        answer = apply_neo4j_query(question)
        
        return Response({"answer": answer})