import torch
import torch.nn as nn
import torch.optim as optim

# 1. Check XPU availability
if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"✓ Intel XPU is available!")
    print(f"  Device count: {torch.xpu.device_count()}")
    print(f"  Device name: {torch.xpu.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("✗ Intel XPU not available, using CPU instead")


# 2. Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 3. Move model to XPU
model = SimpleNet().to(device)
print(f"Model device: {next(model.parameters()).device}")

# 4. Create dummy data and move to XPU
batch_size = 32
input_data = torch.randn(batch_size, 784).to(device)
target = torch.randint(0, 10, (batch_size,)).to(device)

# 5. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 6. Training step
model.train()
optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print(f"\nTraining completed!")
print(f"Loss: {loss.item():.4f}")

# 7. Inference example
model.eval()

test_input = torch.randn(5, 784).to(device)
predictions = model(test_input)
print(f"\nInference completed!")
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions device: {predictions.device}")