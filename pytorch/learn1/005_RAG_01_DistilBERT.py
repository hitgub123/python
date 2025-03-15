import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 超参数
embedding_dim = 768  # DistilBERT 输出维度
batch_size = 3  # 小批量，CPU 友好
max_length = 20  # 短序列，减少内存(max_length = 20)
num_labels = 3  # 假设 3 个回答类别（简化）
device = torch.device("cpu")

# 模拟文档数据集（FAQ）
documents = [
    {"question": "什么是人工智能？", "answer": "人工智能是模拟人类智能的技术。"},
    {"question": "机器学习怎么工作？", "answer": "机器学习通过数据训练模型。"},
    {"question": "深度学习有什么用？", "answer": "深度学习用于图像和语言处理。"},
]

# 加载 DistilBERT（用于检索嵌入）
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()


# 生成文档嵌入
def get_embedding(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # CLS 向量


doc_embeddings = [get_embedding(doc["question"]) for doc in documents]


# 检索函数
def retrieve(query, top_k=1):
    query_emb = get_embedding(query)
    similarities = cosine_similarity([query_emb], doc_embeddings)[0]
    top_idx = np.argsort(similarities)[-top_k:][::-1]
    return [documents[idx]["answer"] for idx in top_idx]


# 生成模型（DistilBERT for Sequence Classification）
class RAGGenerator(nn.Module):
    def __init__(self, num_labels):
        super(RAGGenerator, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        # 冻结大部分层，只微调分类头
        for param in self.distilbert.distilbert.parameters():
            param.requires_grad = False  # 冻结 Transformer 层

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # [batch_size, num_labels]


# 数据准备
class RAGDataset(Dataset):
    def __init__(self, queries, answers):
        self.queries = queries
        self.answers = answers

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        retrieved = retrieve(query)[0]  # 检索到的答案
        inputs = tokenizer(
            retrieved,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].squeeze().to(device)
        attention_mask = inputs["attention_mask"].squeeze().to(device)
        target = torch.tensor(idx, dtype=torch.long).to(
            device
        )  # 用索引作为标签（简化）
        return input_ids, attention_mask, target


# 训练数据
queries = ["人工智能是什么？", "机器学习如何运作？", "深度学习的应用有哪些？"]
answers = [
    "人工智能是模拟人类智能的技术。",
    "机器学习通过数据训练模型。",
    "深度学习用于图像和语言处理。",
]
dataset = RAGDataset(queries, answers)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
generator = RAGGenerator(num_labels).to(device)
optimizer = torch.optim.Adam(generator.parameters(), lr=2e-5)  # 小学习率
criterion = nn.CrossEntropyLoss()
# todo CrossEntropyLoss对照的是非独热编码


# 训练
def train(model, dataloader, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, targets in dataloader:
            input_ids, attention_mask, targets = (
                input_ids.to(device),
                attention_mask.to(device),
                targets.to(device),
            )
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)  # [batch_size, num_labels]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# 运行训练
train(generator, dataloader)


# 测试 RAG
def rag_demo(query):
    retrieved = retrieve(query)[0]
    print(f"Retrieved: {retrieved}")
    inputs = tokenizer(
        retrieved,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        outputs = generator(inputs["input_ids"], inputs["attention_mask"])
        pred_label = torch.argmax(outputs, dim=-1).item()
        print(f"Generated Answer: {answers[pred_label]}")


# 测试
rag_demo("人工智能是什么？")
rag_demo("什么是人工智能？")
rag_demo("机器学习的原理是什么")
rag_demo("深度学习有什么用")
