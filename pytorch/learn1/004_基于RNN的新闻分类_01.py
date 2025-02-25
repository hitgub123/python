import torch, numpy as np, random, pandas as pd, model_util, re
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 超参数
vocab_size = 5000  # 词汇表大小
embedding_dim = 100  # 词嵌入维度
hidden_dim = 128  # RNN 隐藏层维度
output_dim = 4  # 类别数（假设 4 类新闻）
max_len = 20  # 最大序列长度(20)
batch_size = 32
epochs = 100
lr = 0.001
device = torch.device("cpu")  # 用 CPU

text_col = 1
train_data_label = pd.read_csv(
    "pytorch/learn1/datas/AG_NEWS/train.csv", header=None
).to_numpy()
train_data, train_label = train_data_label[:, text_col], train_data_label[:, 0] - 1

test_data_label = pd.read_csv(
    "pytorch/learn1/datas/AG_NEWS/test.csv", header=None
).to_numpy()
test_data, test_label = test_data_label[:, text_col], test_data_label[:, 0] - 1
test_news_lst = test_data_label[:, text_col].copy()


# 数据预处理
def build_vocab(texts, max_size=vocab_size):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    common_words = word_counts.most_common(max_size - 2)  # 留出 <PAD> 和 <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in common_words:
        if re.fullmatch(r"[a-zA-Z0-9]+", word):
            vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab, max_len=max_len):
    seq = [vocab.get(word, 1) for word in text.split()]  # <UNK> 为 1
    seq = seq[:max_len] + [0] * (max_len - len(seq))  # 填充 <PAD>
    return seq


# 数据准备
vocab = build_vocab(train_data)
for i in range(train_data.shape[0]):
    train_data[i] = np.array(text_to_sequence(train_data[i], vocab))

for i in range(test_data.shape[0]):
    test_data[i] = np.array(text_to_sequence(test_data[i], vocab))


def handle_data(data, label):
    data = np.concatenate(data).astype(np.int64).reshape(data.shape[0], -1)
    label = label.astype(np.int64)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return [(data[i], label[i]) for i in range(data.shape[0])]

train_data_label_handled = handle_data(train_data, train_label)
test_data_label_handled = handle_data(test_data, test_label)

train_loader = DataLoader(train_data_label_handled, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data_label_handled, batch_size=batch_size)

# RNN 模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: [batch_size, max_len]
        embedded = self.embedding(text)  # [batch_size, max_len, embedding_dim]
        output, hidden = self.rnn(embedded)  # hidden: [1, batch_size, hidden_dim]
        return self.fc(hidden.squeeze(0))  # [batch_size, output_dim]


# 初始化模型
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# 测试函数
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 训练循环
model_name = model_util.get_model_name(__file__)
max_acc = model_util.load_model(model, model_name)
print("max_acc", max_acc)
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)
    print(
        f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}"
    )
    if max_acc < test_acc:
        max_acc = test_acc
        model_util.save_model(model, model_name, max_acc)


# 测试单条新闻
# 1,World：有关国际新闻和全球事件的报道。
# 2,Sports：涵盖体育新闻和事件的文章。
# 3,Business：与商业、经济和金融相关的报道。
# 4,Science/Technology：涉及科学和技术领域的新闻。
model.eval()
classes = ["World", "Sports", "Business", "Science/Technology"]

for i in range(10):
    test_news = test_news_lst[i]
    seq = test_data[i].to(device)
    actual = test_label[i]
    with torch.no_grad():
        pred = model(seq)
        label = torch.argmax(pred).item()
        print(f"News: {test_news}, Predicted: {classes[label]},Result: {actual==label}")
