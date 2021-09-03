#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gumbel_sigmoid_softmax import gumbel_sigmoid
import torch
import numpy as np


# ### Simple demo
# * Sample from gumbel-softmax
# * Average over samples

# In[ ]:


temperature = 0.1
logits = np.linspace(-5, 5, 10).reshape([1,-1])
logits = torch.Tensor(logits)
gumbel_sigm = gumbel_sigmoid(logits, temperature=temperature)
sigm = torch.sigmoid(logits)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title('gumbel-sigmoid samples')
for i in range(10):
    plt.plot(range(10), gumbel_sigmoid(logits, temperature=temperature)[0], marker='o', alpha=0.25)
plt.ylim(0,1)
plt.show()

plt.title('average over samples')
samples = torch.stack(
    [gumbel_sigmoid(logits, temperature=temperature)[0] for _ in range(500)],
    dim=0
)
# samples
plt.plot(range(10), torch.mean(samples, axis=0),
         marker='o', label='gumbel-sigmoid average')

plt.plot(sigm[0], marker='+',label='regular softmax')
plt.legend(loc='best')


# # Autoencoder with gumbel-sigmoid
# 
# * We do not use any bayesian regularization, simply optimizer by backprop
# * Hidden layer contains 32 units

# In[ ]:


from sklearn.datasets import load_digits
X = load_digits().data
print(X.shape)


# In[ ]:


plt.matshow(X[1111].reshape(8,8))


# In[ ]:


# 设备配置
torch.cuda.set_device(0) # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


import torch
from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 32)  # #bottleneck
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64)
        )
        self.tau = 1.0

    def set_temperature(self, tau):
        self.tau = tau
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, x, hard=False):
        return gumbel_sigmoid(x, temperature=self.tau, hard=hard)

    def forward(self, x, hard=False):
        x = self.encode(x)
        x = self.reparameterization(x, hard=hard)
        x = self.decode(x)
        return x

model = EncoderDecoder().to(device)
print(model)


# ## Training loop
# * We gradually reduce temperature from 1 to 0.01 over time

# In[ ]:


tau_values = np.logspace(0,-2,20000)
tau_values = np.clip(tau_values, 0.01, None)
plt.plot(tau_values)


# In[ ]:


criteria = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for i, t in enumerate(tau_values):
    batch = X[np.random.choice(len(X),32)]
    batch = torch.Tensor(batch).to(device)
    model.set_temperature(t)
    output = model(batch)
    loss = criteria(batch, output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        with torch.no_grad():
            batch = X[np.random.choice(len(X), 320)]
            batch = torch.Tensor(batch).to(device)
            output = model(batch)
            loss = criteria(batch, output)
            print('%.3f' % loss, end='\t', flush=True)


# In[ ]:


#functions for visualization
get_sample = lambda x : model(x.to(device)).detach().cpu().numpy()
get_sample_hard = lambda x : model(x.to(device), hard=True).detach().cpu().numpy()
get_code = lambda x : model.reparameterization(model.encode(x.to(device))).detach().cpu().numpy()


# In[ ]:


for i in range(10):
    X_sample = X[np.random.randint(len(X)),None,:]
    X_sample = torch.Tensor(X_sample).unsqueeze(0)
    plt.figure(figsize=[12,4])
    plt.subplot(1,4,1)
    plt.title("original")
    plt.imshow(X_sample.reshape([8,8]),interpolation='none',cmap='gray')
    plt.subplot(1,4,2)
    plt.title("gumbel")
    plt.imshow(get_sample(X_sample).reshape([8,8]),interpolation='none',cmap='gray')
    plt.subplot(1,4,3)
    plt.title("hard-max")
    plt.imshow(get_sample_hard(X_sample).reshape([8,8]),interpolation='none',cmap='gray')
    plt.subplot(1,4,4)
    plt.title("code")
    plt.imshow(get_code(X_sample).reshape(8,4),interpolation='none',cmap='gray')
    plt.show()


# In[ ]:




