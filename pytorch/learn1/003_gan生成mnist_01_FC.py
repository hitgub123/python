import torch, model_util, sys
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 超参数
latent_dim = 100  # 噪声向量的维度
hidden_dim = 256  # 隐藏层维度
image_dim = 28 * 28  # MNIST 图像展平后的大小 (784)
batch_size = 64  # 批量大小
epochs = 50  # 训练轮数
lr = 0.0002  # 学习率
device = torch.device("cpu")  # 用 CPU

# 数据加载和预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]  # 归一化到 [-1, 1]
)
mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh(),  # 输出范围 [-1, 1]，匹配归一化后的图像
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)  # 重塑为图像形状 [batch_size, 1, 28, 28]


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # 输出概率 [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(-1, image_dim)  # 展平图像
        validity = self.model(img_flat)
        return validity


# 初始化模型和优化器
generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()  # 二元交叉熵损失


# 训练函数
def train_gan():
    model_name_g = model_util.get_model_name(__file__, "g")
    V_g = model_util.load_model(generator, model_name_g)
    V_g = sys.maxsize if V_g == 0 else V_g
    model_name_d = model_util.get_model_name(__file__, "d")
    V_d = model_util.load_model(discriminator, model_name_d)
    V_d = sys.maxsize if V_d == 0 else V_d
    for epoch in range(epochs):
        g_loss_sum, d_loss_sum = 0, 0
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 标签：真实图像为 1，生成图像为 0
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            d_optimizer.zero_grad()
            d_real_loss = criterion(
                discriminator(real_imgs), real_label
            )  # 真实图像损失

            z = torch.randn(batch_size, latent_dim).to(device)  # 随机噪声
            fake_imgs = generator(z)
            # detach方法：避免计算generator的梯度，减少计算量。
            # 如果去掉detach，d_optimizer只绑定了判别器的参数，所以d_optimizer.step()不会更新生成器的参数，
            # 但是生成器参数的梯度会被计算，然后在g_optimizer.zero_grad()被清除。所以不会影响生成器的训练。
            # 如果没有g_optimizer.zero_grad()，这里的梯度会和g_loss处的梯度累计起来更新生成器的参数。
            d_fake_loss = criterion(
                discriminator(fake_imgs.detach()), fake_label
            )  # 生成图像损失

            d_loss = d_real_loss + d_fake_loss
            d_loss_sum += d_loss.item() * batch_size
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            g_loss = criterion(
                discriminator(fake_imgs), real_label
            )  # 目标：让判别器认为假图是真的
            g_loss_sum += g_loss.item() * batch_size
            g_loss.backward()
            g_optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                    f"g_loss: {g_loss.item():.4f} d_loss: {d_loss.item():.4f}"
                )

        g_loss_avg, d_loss_avg = g_loss_sum / len(mnist), d_loss_sum / len(mnist)
        print("g_loss_avg=", g_loss_avg, "d_loss_avg=", d_loss_avg)
        print("V_g=", V_g, "V_d=", V_d)
        if g_loss_avg < V_g:
            V_g = g_loss_avg
            model_util.save_model(generator, model_name_g, g_loss_avg)
        if d_loss_avg < V_d:
            V_d = d_loss_avg
            model_util.save_model(discriminator, model_name_d, d_loss_avg)

        # 每 10 个 epoch 保存生成的图像
        if epoch % 10 == 0:
            with torch.no_grad():
                fake_imgs = generator(torch.randn(16, latent_dim).to(device)).cpu()
                show_images(fake_imgs, epoch)


# 显示生成的图像
def show_images(images, epoch):
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    images = images * 0.5 + 0.5  # 反归一化到 [0, 1]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"gan_output_epoch_{epoch}.png")
    plt.close()


# 开始训练
if __name__ == "__main__":
    train_gan()

    # 生成并展示最终结果
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        final_imgs = generator(z).cpu()
        show_images(final_imgs, "Final")
