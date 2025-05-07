import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=25):
        super(Autoencoder, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """仅使用编码器部分，提取特征"""
        return self.encoder(x)

    # 在autoencoder.py中修改损失函数计算
    def train_model(self, train_loader, epochs=100, learning_rate=0.3, l2_weight=0.0059,
                    sparsity_param=0.002, sparsity_reg=0.007, device='cuda', momentum=0.2):
        self.to(device)
        self.train()

        # 论文使用的是Scaled Conjugate Gradient
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_weight)

        for epoch in range(epochs):
            total_loss = 0
            reconstruction_losses = 0
            reg_losses = 0
            sparsity_losses = 0

            for data, _ in train_loader:
                data = data.to(device)

                # 前向传播
                outputs = self(data)
                encoded_output = self.encoder(data)

                # 1. 计算重建损失 (MSE)
                reconstruction_loss = torch.mean((outputs - data) ** 2)

                # 2. 计算稀疏性约束 - 公式(8)中的KL散度
                # 计算平均激活值
                avg_activation = torch.mean(encoded_output, dim=0)

                # 计算KL散度
                # KL(p||p') = p*log(p/p') + (1-p)*log((1-p)/(1-p'))
                kl_divergence = sparsity_param * torch.log(sparsity_param / (avg_activation + 1e-10)) + \
                                (1 - sparsity_param) * torch.log((1 - sparsity_param) / (1 - avg_activation + 1e-10))
                kl_divergence = torch.sum(kl_divergence)

                # 3. 总损失 = 重建损失 + 稀疏性正则化 * KL散度 (L2正则化已在优化器中)
                loss = reconstruction_loss + sparsity_reg * kl_divergence

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 累计损失
                total_loss += loss.item()
                reconstruction_losses += reconstruction_loss.item()
                sparsity_losses += (sparsity_reg * kl_divergence).item()

            # 输出详细训练信息
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.6f}, '
                      f'Recon Loss: {reconstruction_losses / len(train_loader):.6f}, '
                      f'Sparsity Loss: {sparsity_losses / len(train_loader):.6f}')