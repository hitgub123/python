import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 超参数
embedding_dim = 768  # DistilBERT 用于检索
batch_size = 1
max_length = 20  # 输入和生成的最大长度
device = torch.device("cpu")

# 数据集
documents = [
    {"question": "什么是人工智能？", "answer": "人工智能是模拟人类智能的技术。"},
    {"question": "机器学习怎么工作？", "answer": "机器学习通过数据训练模型。"},
    {"question": "深度学习有什么用？", "answer": "深度学习用于图像和语言处理。"},
]

queries = ["人工智能是什么？", "机器学习如何运作？", "深度学习能做什么？"]
answers = [
    "人工智能是模拟人类智能的技术。",
    "机器学习通过数据训练模型。",
    "深度学习用于图像和语言处理。",
]

# 加载模型和分词器
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

# 为 GPT2 添加 pad token（DistilGPT2 默认无 padding）
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id


# 生成嵌入（用于检索）
def get_embedding(text):
    inputs = bert_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


doc_embeddings = [get_embedding(doc["question"]) for doc in documents]


# 检索函数
def retrieve(query, top_k=1):
    query_emb = get_embedding(query)
    similarities = cosine_similarity([query_emb], doc_embeddings)[0]
    top_idx = np.argsort(similarities)[-top_k:][::-1]
    return [documents[idx]["answer"] for idx in top_idx]


# 数据集
class RAGDataset(Dataset):
    def __init__(self, queries, answers):
        self.queries = queries
        self.answers = answers

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        retrieved = retrieve(query)[0]
        # 输入：检索到的答案作为提示
        input_text = f"回答: {retrieved}"
        inputs = gpt2_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        input_ids = inputs["input_ids"].squeeze().to(device)
        attention_mask = inputs["attention_mask"].squeeze().to(device)
        # 目标：完整的答案
        target_ids = (
            gpt2_tokenizer(
                self.answers[idx],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )["input_ids"]
            .squeeze()
            .to(device)
        )
        return input_ids, attention_mask, target_ids


# 数据准备
dataset = RAGDataset(queries, answers)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 冻结部分层（优化 CPU）
# for param in gpt2_model.transformer.parameters():
#     param.requires_grad = False  # 冻结 Transformer 层，只训输出层
# optimizer = torch.optim.Adam(gpt2_model.parameters(), lr=2e-5)

# 冻结 Transformer 层
for param in gpt2_model.transformer.parameters():
    param.requires_grad = False  # 只冻结 Transformer，输出层仍可训练
for param in gpt2_model.lm_head.parameters():
    param.requires_grad = True  # 确保输出层可训练
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, gpt2_model.parameters()), lr=2e-5
)

criterion = nn.CrossEntropyLoss(ignore_index=gpt2_tokenizer.pad_token_id)


# 训练
def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, target_ids in dataloader:
            input_ids, attention_mask, target_ids = (
                input_ids.to(device),
                attention_mask.to(device),
                target_ids.to(device),
            )
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=target_ids
            )
            loss = outputs.loss  # GPT2 自带损失计算
            # 调试：检查损失
            if loss is None or not loss.requires_grad:
                print("Loss is None or does not require grad!")
                print(f"Outputs: {outputs.logits}")
                continue
            # print(input_ids.requires_grad)  # 应为 False（输入不需要梯度）
            # print(outputs.logits.requires_grad)  # 应为 True（输出需梯度）
            # print(outputs.loss is not None)  # 应为 True（损失存在）
            # print(outputs.loss.requires_grad)  # 应为 True（损失需梯度）
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# 运行训练
train(gpt2_model, dataloader)


# 测试
def rag_demo(query):
    retrieved = retrieve(query)[0]
    print(f"Retrieved: {retrieved}")
    input_text = f"回答: {retrieved}"
    inputs = gpt2_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        outputs = gpt2_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length + 10,  # 允许生成稍长于输入
            num_beams=2,  # 束搜索提升质量
            early_stopping=True,
        )
    generated = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Answer: {generated}")


# 测试
rag_demo("人工智能是什么？")
rag_demo("什么是人工智能？")
rag_demo("机器学习的原理是什么")
rag_demo("深度学习有什么用")
