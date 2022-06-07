import torch
from matplotlib import pyplot
from torch import nn
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
    def forward(self, x):
        x = self.generator(x)
        return self.discriminator(x)
    
def generate_real_samples(n):
    X1 = torch.rand(n) - 0.5
    X1=X1.to(device)
    X2 = X1 * X1# given thing
    X2=X2.to(device)
    X = torch.hstack((X1.reshape(n, 1), X2.reshape(n, 1)))
    X=X.to(device)
    return X, torch.ones((n, 1))
 
def generate_latent_points(latent_dim, n):
    x_input = torch.randn(latent_dim * n).reshape(n, latent_dim)
    x_input=x_input.to(device)
    return x_input
 
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    x_input=x_input.to(device)
    X=generator(x_input)
    # print(torch.sin(X))
    # X = generator(x_input) - 0.5
    return X, torch.zeros((n, 1))
 
def summarize_performance(epoch, generator, discriminator, latent_dim, n=1000):
    
    x, y = generate_real_samples(n)
    x=x.to(device)
    y=y.to(device)
    output = discriminator(x)
    output=output.to(device)
    acc_real = (output == y).float().sum()/n
    x2, y2 = generate_fake_samples(generator, latent_dim, n)
    x2=x2.to(device)
    y2=y2.to(device)
    output = discriminator(x2)
    output=output.to(device)
    acc_fake = (output == y2).float().sum()/n
    print(epoch, acc_real, acc_fake)
    x = x.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    pyplot.scatter(x[:, 0], x[:, 1], color='red')
    pyplot.scatter(x2[:, 0], x2[:, 1], color='blue')
    pyplot.savefig('gan_%d.png' % epoch)
    pyplot.close()
 
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
    # process the model over GPU
    gan_model.to(device)    
    g_model.to(device)
    d_model.to(device)
   
    optimizer_gan = torch.optim.Adam(gan_model.generator.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)

    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        
        x, y = generate_real_samples(half_batch)
        x=x.to(device)
        y=y.to(device)
        loss = criterion(d_model(x), y)
        d_model.zero_grad()
        loss.backward()
        optimizer_d.step()
        x2, y2 = generate_fake_samples(g_model, latent_dim, half_batch)
        x2=x2.to(device)
        y2=y2.to(device)
        loss2 =  criterion(d_model(x2), y2)
        d_model.zero_grad()
        loss2.backward()
        optimizer_d.step()
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = torch.ones((n_batch, 1))
        x_gan=x_gan.to(device)
        y_gan=y_gan.to(device)
        loss = criterion(gan_model(x_gan), y_gan)
        gan_model.zero_grad()
        loss.backward()
        optimizer_gan.step()

        if (i+1) % n_eval == 0:
            summarize_performance(i+1, g_model, d_model, latent_dim)

if __name__ == '__main__':  
    n_inputs = 2
    latent_dim =32
    h_dim = 256
    discriminator = nn.Sequential(nn.Linear(n_inputs, h_dim), nn.LeakyReLU(0.1), nn.Linear(h_dim, 1), nn.Sigmoid())
    
    generator = nn.Sequential(nn.Linear(latent_dim, h_dim), nn.LeakyReLU(0.1), nn.Linear(h_dim, n_inputs), nn.Tanh())
    gan_model = GAN(generator, discriminator)
    train(generator, discriminator, gan_model, latent_dim)
