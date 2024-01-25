import torch
import matplotlib.pyplot as plt

error_rates = torch.load('error_rates.pth')
print(error_rates)
print('average error rate:', torch.mean(error_rates))

losses = torch.load('losses.pth')
plt.plot(losses)
plt.xlabel('batch')
plt.ylabel('loss')
plt.yscale('log')
plt.savefig('losses.pdf')
plt.savefig('losses.png')
plt.show()

