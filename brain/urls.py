from django.urls import path
from .views import ExtractTestView, AskKnowledgeView


urlpatterns = [
    path('extract/', ExtractTestView.as_view(), name='extract_relations'),
    path('ask/', AskKnowledgeView.as_view(), name='apply_neo4j_query'), 
]