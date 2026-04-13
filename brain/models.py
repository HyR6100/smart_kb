from django.db import models

# Create your models here.
class Document4pdf(models.Model):
    """存储上传的pdf"""
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    processd = models.BooleanField(default=False) # True表示已经在Neo4j中
    created_at = models.DateTimeField(auto_now_add=True) # 创造时间自动生成

class Conversation4collection(models.Model):
    """微调时候收集的对话数据"""
    question = models.TextField()
    answer = models.TextField()
    
    good_answer = models.TextField(blank=True, null=True)
    is_good_sample = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)