#%%
import torch
import torch.optim as optim
from torch import nn


import statistics
import numpy as np

import matplotlib.pyplot as plt

from utils import ema, load_pickle, save_pickle
from rose_models import Discriminator, Generator
from rose_plots import coloured_plt, plot_2d
from rose_data import generate_noise, sample_from_target_function, sample_noise, target_function

from time import time
#%%
def V_GAN(
    generator=Generator(),
    discriminator=Discriminator(),
    noise_fn=sample_noise,
    data_fn=sample_from_target_function,
    plot_every = 0, 
    lr_d=1e-3, #1e-3
    lr_g=2e-4, #2e-4 , MC:1e-3
    betas_adam=(0.9, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
    epochs=100, 
    batches=100, 
    batch_size=32,
    gen_size=4000,
    device='cpu'):
    """ Vanilla GAN builder and trainer
    Args:
        generator: torch.nn.Module
        discriminator: torch.nn.Module
        noise_fn: function that returns a tensor of noise
        data_fn: function that returns a tensor of data
        plot_every: int, plot every n epochs
        lr_d: float, learning rate for discriminator
        lr_g: float, learning rate for generator
        betas_adam: tuple, betas for Adam optimizer
        epochs: int, number of epochs to train for
        batches: int, number of batches per epoch
        batch_size: int, batch size
        gen_size: int, number of samples to generate for plotting
        device: torch.device, device to train on
    Returns:
    A dictionary with the following keys:
        losses: tuple, (loss_g, loss_d_real, loss_d_fake)
            loss_g: list, generator loss
            loss_d_real: list, discriminator loss on real data
            loss_d_fake: list, discriminator loss on fake data
        gen_samples: list, generated samples
        latent_samples: list, latent samples
    """

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion = nn.BCELoss()
    optim_d = optim.Adam(discriminator.parameters(),
                                lr=lr_d, betas=betas_adam)
    optim_g = optim.Adam(generator.parameters(),
                                lr=lr_g, betas=betas_adam)
    target_ones = torch.ones((batch_size, 1)).to(device)
    target_zeros = torch.zeros((batch_size, 1)).to(device)
    loss_g, loss_d_real, loss_d_fake = [], [], []
    gen_samples, latent_samples = [], []
    start = time()
    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch in range(batches):

            # Train the generator one step
            generator.zero_grad()
            gen_latent_vec = noise_fn(batch_size)
            generated = generator(gen_latent_vec)
            classifications = discriminator(generated)
            loss = criterion(classifications, target_ones)
            loss.backward()
            optim_g.step()
            lg_ = loss.item()

            # Train the discriminator one step
            discriminator.zero_grad()

            # real samples
            real_samples = data_fn(batch_size)
            pred_real = discriminator(real_samples)
            loss_real = criterion(pred_real, target_ones)

            # generated samples
            gen_latent_vec = noise_fn(batch_size)
            with torch.no_grad():
                fake_samples = generator(gen_latent_vec)
            pred_fake = discriminator(fake_samples)
            loss_fake = criterion(pred_fake, target_zeros)

            # combine
            loss = (loss_real + loss_fake) / 2
            loss.backward()
            optim_d.step()

            ldr_ = loss_real.item()
            ldf_ = loss_fake.item()

            loss_g_running += lg_
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_

        loss_g.append(loss_g_running / batches)
        loss_d_real.append(loss_d_real_running / batches)
        loss_d_fake.append(loss_d_fake_running / batches)

        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
                f" G={loss_g[-1]:.3f},"
                f" D(x)={loss_d_real[-1]:.3f},"
                f" D(G(z))={loss_d_fake[-1]:.3f}")

        def generate_sample(gen_size):
            gen_latent_vec = noise_fn(gen_size)
            gen_sample = generator(gen_latent_vec).detach().cpu().numpy()
            return gen_sample, gen_latent_vec

        gen_sample, gen_latent_vec = generate_sample(gen_size)
        gen_samples.append(gen_sample)
        latent_samples.append(gen_latent_vec)

        if plot_every != 0 and (epoch+1) % plot_every == 0:
            plot_2d(gen_sample,plot_target=True, epoch="at Epoch "+str(epoch+1))
    
    model_dict ={
        "latent_samples": latent_samples,
        "gen_samples": gen_samples,
        "losses": (loss_g, loss_d_real, loss_d_fake)
    }

    return model_dict
# %%
mc_model = V_GAN(
    generator=Generator(layer_size=[2,512,512,512,2], layer_activation=nn.LeakyReLU(0.1)),
    discriminator=Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1)),
    noise_fn=sample_noise,
    data_fn=sample_from_target_function,
    plot_every = 1, 
    lr_d=1e-3, #1e-3
    lr_g=3e-4, #2e-4 , MC:1e-3
    betas_adam=(0.9, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
    epochs=100, # pretend 500
    batches=1,  # pretend 100
    batch_size=128,
    gen_size=4000,
    device='cpu')

save_pickle('mc_exp',mc_model)
# %%
mc_exp_l = load_pickle('mc_exp')

# %%
for epoch in range(0,100,2):
    # plot_2d(mc_exp_l['gen_samples'][epoch], plot_target=True, save_fig=f'2d_mc_{epoch}.pdf')
    plot_2d(mc_exp_l['gen_samples'][epoch], plot_target=True, epoch=f'{epoch}')
# %%
for epoch in [0,20,40,60,80]:
    plot_2d(mc_exp_l['gen_samples'][epoch], plot_target=True, save_fig=f'2d_mc_{epoch}.pdf')
    # plot_2d(mc_exp_l['gen_samples'][epoch], plot_target=True)

# %%
mc2_model = V_GAN(
    generator=Generator(layer_size=[2,512,512,512,2], layer_activation=nn.LeakyReLU(0.1)),
    discriminator=Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1)),
    noise_fn=sample_noise,
    data_fn=sample_from_target_function,
    plot_every = 10, 
    lr_d=1e-3, #1e-3
    lr_g=2e-4, #2e-4 , MC:1e-3
    betas_adam=(0.5, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
    epochs=600, 
    batches=100, 
    batch_size=128,
    gen_size=4000,
    device='cpu')
# %%
save_pickle('mc2_exp',mc2_model)
# %%
mc2_exp_l = load_pickle('mc2_exp')

# %%
for epoch in range(0,600,99):
    # plot_2d(mc_exp_l['gen_samples'][epoch], plot_target=True, save_fig=f'2d_mc_{epoch+1}.pdf')
    plot_2d(mc2_exp_l['gen_samples'][epoch], plot_target=True, epoch=f'{epoch}')
# %%
for epoch in [49, 449, 559, 579, 599]:
    # plot_2d(mc2_exp_l['gen_samples'][epoch], plot_target=True, save_fig=f'2d_mc2_{epoch+1}.pdf')
    plot_2d(mc2_exp_l['gen_samples'][epoch], plot_target=True)

# %%
lr_exp = V_GAN(
    generator=Generator(layer_size=[2,512,512,512,2], layer_activation=nn.LeakyReLU(0.1)),
    discriminator=Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1)),
    noise_fn=sample_noise,
    data_fn=sample_from_target_function,
    plot_every = 10, 
    lr_d=1e-3, #1e-3
    lr_g=2e-4, #2e-4 , MC:1e-3
    betas_adam=(0.5, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
    epochs=600, 
    batches=100, 
    batch_size=128,
    gen_size=4000,
    device='cpu')

#%%
# gan_lr_experiments = load_pickle('lr_exp')
gen_lr_exp = {}
exp_name_dict = gen_lr_exp
for lr in [0.0003, 0.0002, 0.00018, 0.00015, 0.0001]:
    if str(lr) not in exp_name_dict.keys():
        print(f"Learning Rate {lr} not found, running experiement")
        name_exp = V_GAN(
            generator=Generator(layer_size=[2,512,512,512,2], layer_activation=nn.LeakyReLU(0.1)),
            discriminator=Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1)),
            noise_fn=sample_noise,
            data_fn=sample_from_target_function,
            plot_every = 50, 
            lr_d=1e-3, #1e-3
            lr_g=lr, #2e-4 , MC:1e-3
            betas_adam=(0.5, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
            epochs=600, 
            batches=100, 
            batch_size=128,
            gen_size=2000,
            device='cpu')
        exp_name_dict[str(lr)] = name_exp
save_pickle('gen_lr_exp', exp_name_dict)

#%%
gen_lr_exp = load_pickle('gen_lr_exp')
# %%
for lr in [0.0003, 0.0002, 0.00018, 0.00015, 0.0001]:
    for epoch in [599]:
        # plot_2d(mc2_exp_l['gen_samples'][epoch], plot_target=True, save_fig=f'2d_mc2_{epoch+1}.pdf')
        plot_2d(gen_lr_exp[str(lr)]['gen_samples'][epoch], plot_target=True)

metric_names = {
    'g-loss':'loss [G(z)]',
    'df-loss':'loss [D(G(z)]',
    'dr-loss': 'loss [D(x)]',
    'd-loss': 'loss 0.5*[D(G(z)+D(x)]',
    'd-loss-ema': 'loss (EMA)',
    'df-loss-ema': 'loss (EMA)',
    'g-loss-ema': 'loss (EMA)'
}
def get_metric(metric, exp_dict, exp_var):
    if metric == 'dr-loss':
        _, p_metric, _ = exp_dict[str(exp_var)]['losses']
        metric = 'D(x)'
    elif metric == 'df-loss':
        *_, p_metric = exp_dict[str(exp_var)]['losses']
        metric = 'D(G(z))'
    elif metric == 'dr-loss':
        _, p_metric, _ = exp_dict[str(exp_var)]['losses']
        metric = 'D(x)'
    elif metric == 'd-loss':
        p_metric = [statistics.mean(k) for k in zip(get_metric('dr-loss', exp_dict, exp_var),get_metric('df-loss', exp_dict, exp_var))]
        metric = 'Average Discriminator Loss'
    elif metric == 'd-loss-ema':
        p_metric = ema(get_metric('d-loss',exp_dict, exp_var),alpha=0.7)
    elif metric == 'df-loss-ema':
        p_metric = ema(get_metric('df-loss',exp_dict, exp_var),alpha=0.7)
    elif metric == 'g-loss':
        p_metric, *_ = exp_dict[str(exp_var)]['losses']
        metric = 'G(z)'
    elif metric == 'g-loss-ema':
        p_metric = ema(get_metric('g-loss',exp_dict, exp_var),alpha=0.8)
    else:
        print('must be one of d-value, w-value, mean, std, d-loss, g-loss')
    return p_metric

def plot_metric(metric, exp_dicts, var_to_plot=[], fig_=plt.figure(), plt_config=(True,True,'',1)):
    l1, l2, l_title, plot_every = plt_config
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ["-","--"]
    ax_ = fig_.add_subplot(111)
    for j, exp_dict in enumerate(list(exp_dicts.values())) :
        if not var_to_plot: var_to_plot = [int(x) if int(float(x)) == float(x) else float(x) for x in list(exp_dict.keys())] 
        for i, exp_var in enumerate(var_to_plot):
            p_metric = get_metric(metric, exp_dict,exp_var)
            ax_.plot(np.arange(1, len(p_metric)+1, plot_every), 
                        p_metric[::plot_every], label=f'{exp_var:.1E}',color=color_cycle[i], linestyle=linestyle_cycle[j])
    
    lines = ax_.get_lines()
    # if l1: legend_1 = ax_.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]], title=l_title, ncol=len(lines)+1)
    if l1: legend_1 = ax_.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]], ncol=len(lines))
    if l2: legend_2 = ax_.legend(lines[::i+1], list(exp_dicts.keys()))
    if l1 and l2: ax_.add_artist(legend_1)
    
    ax_.set_xlabel('Epoch Number')
    ax_.set_ylabel(metric_names[metric] if metric in metric_names.keys() else metric)

    return ax_
#%%
# get_metric
exp_dicts3={
    'Varied Learning Rate': gen_lr_exp
    }
lr_to_plot=[0.0003, 0.0002, 0.00018, 0.00015, 0.0001]
#%%
fig_x = plt.figure(figsize=(7.4, 2))
ax_x = plot_metric('g-loss-ema',exp_dicts3, var_to_plot=lr_to_plot,fig_=fig_x, plt_config=(True, False, 'Learning Rate', 4))
# ax_x.set_ylim(0.685,0.695)
ax_x.set_ylim(0.68,0.8)
fig_lr = ax_x.get_figure()
fig_lr.tight_layout()
fig_lr.savefig('../report/figures/2d_gen_loss_ema.pdf', bbox_inches='tight', pad_inches=0)

#%%
fig_x = plt.figure(figsize=(7.4, 2))
ax_x = plot_metric('d-loss-ema',exp_dicts3, var_to_plot=lr_to_plot,fig_=fig_x, plt_config=(True, False, 'Learning Rate', 4))
ax_x.set_ylim(0.67,0.694)
# ax_x.set_ylim(0.68,0.8)
fig_lr = ax_x.get_figure().tight_layout()
# fig_lr.savefig('../report/figures/2d_gen_loss_ema.pdf', bbox_inches='tight', pad_inches=0)
#%%
fig_x = plt.figure(figsize=(7.4, 2))
ax_x = plot_metric('g-loss',exp_dicts3, var_to_plot=lr_to_plot,fig_=fig_x, plt_config=(True, False, 'Learning Rate'))
ax_x.get_figure().tight_layout()
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('d-loss',exp_dicts3, var_to_plot=lr_to_plot,fig_=fig_x, plt_config=(True, False))
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('d-loss-ema',exp_dicts3, var_to_plot=lr_to_plot,fig_=fig_x, plt_config=(True, False))
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('g-loss',exp_dicts3, var_to_plot=lr_to_plot,fig_=fig_x, plt_config=(True, False))

#%%
def V_GAN_smooth(
    generator=Generator(),
    discriminator=Discriminator(),
    noise_fn=sample_noise,
    data_fn=sample_from_target_function,
    plot_every = 0, 
    lr_d=1e-3, #1e-3
    lr_g=2e-4, #2e-4 , MC:1e-3
    betas_adam=(0.9, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
    epochs=100, 
    batches=100, 
    batch_size=32,
    gen_size=4000,
    device='cpu',
    one_sided = 0.9):
    """
    Vanilla GAN with smooth labels
    Additional Args:
        one_sided: Smoothened target for discriminator
    """

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion = nn.BCELoss()
    optim_d = optim.Adam(discriminator.parameters(),
                                lr=lr_d, betas=betas_adam)
    optim_g = optim.Adam(generator.parameters(),
                                lr=lr_g, betas=betas_adam)
    target_ones = torch.ones((batch_size, 1)).to(device)
    target_smooth = one_sided*torch.ones((batch_size, 1)).to(device) # Smoothened Target
    target_zeros = torch.zeros((batch_size, 1)).to(device)
    loss_g, loss_d_real, loss_d_fake = [], [], []
    gen_samples, latent_samples = [], []
    start = time()
    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        for batch in range(batches):

            # Train the generator one step
            generator.zero_grad()
            gen_latent_vec = noise_fn(batch_size)
            generated = generator(gen_latent_vec)
            classifications = discriminator(generated)
            loss = criterion(classifications, target_ones)
            loss.backward()
            optim_g.step()
            lg_ = loss.item()

            # Train the discriminator one step
            discriminator.zero_grad()

            # real samples
            real_samples = data_fn(batch_size)
            pred_real = discriminator(real_samples)
            loss_real = criterion(pred_real, target_smooth)

            # generated samples
            gen_latent_vec = noise_fn(batch_size)
            with torch.no_grad():
                fake_samples = generator(gen_latent_vec)
            pred_fake = discriminator(fake_samples)
            loss_fake = criterion(pred_fake, target_zeros)

            # combine
            loss = (loss_real + loss_fake) / 2
            loss.backward()
            optim_d.step()

            ldr_ = loss_real.item()
            ldf_ = loss_fake.item()

            loss_g_running += lg_
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_

        loss_g.append(loss_g_running / batches)
        loss_d_real.append(loss_d_real_running / batches)
        loss_d_fake.append(loss_d_fake_running / batches)

        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
                f" G={loss_g[-1]:.3f},"
                f" D(x)={loss_d_real[-1]:.3f},"
                f" D(G(z))={loss_d_fake[-1]:.3f}")

        def generate_sample(gen_size):
            gen_latent_vec = noise_fn(gen_size)
            gen_sample = generator(gen_latent_vec).detach().cpu().numpy()
            return gen_sample, gen_latent_vec

        gen_sample, gen_latent_vec = generate_sample(gen_size)
        gen_samples.append(gen_sample)
        latent_samples.append(gen_latent_vec)

        if plot_every != 0 and (epoch+1) % plot_every == 0:
            plot_2d(gen_sample,plot_target=True, epoch="at Epoch "+str(epoch+1))
    
    model_dict ={
        "latent_samples": latent_samples,
        "gen_samples": gen_samples,
        "losses": (loss_g, loss_d_real, loss_d_fake)
    }

    return model_dict

# %%
smooth_model = V_GAN_smooth(
    generator=Generator(),
    discriminator=Discriminator(),
    noise_fn=sample_noise,
    data_fn=sample_from_target_function,
    plot_every = 10, 
    lr_d=1e-3, #1e-3
    lr_g=1.8e-4, #2e-4 , MC:1e-3
    betas_adam=(0.5, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
    epochs=100, 
    batches=100, 
    batch_size=128,
    gen_size=2000,
    device='cpu',
    one_sided = 0.9)

#%%
# one_sided_exp = {}
one_sided_exp = load_pickle('one_sided_exp')
exp_name_dict = one_sided_exp
for exp_var in [0.95,0.9,0.8,0.7,0.6]:
    if str(exp_var) not in exp_name_dict.keys():
        print(f"Experiment Variable {exp_var} not found, running experiement")
        name_exp = V_GAN_smooth(
            generator=Generator(),
            discriminator=Discriminator(),
            noise_fn=sample_noise,
            data_fn=sample_from_target_function,
            plot_every = 50, 
            lr_d=1e-3, #1e-3
            lr_g=1.8e-4, #2e-4 , MC:1e-3
            betas_adam=(0.5, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
            epochs=600, 
            batches=100, 
            batch_size=128,
            gen_size=2000,
            device='cpu',
            one_sided = exp_var)
        exp_name_dict[str(exp_var)] = name_exp
save_pickle('one_sided_exp', exp_name_dict)

#%%
os_exp = load_pickle('one_sided_exp')
# %%
# Print Losses
for lam in [0.95,0.9,0.8,0.7,0.6]:
    loss_g, loss_d_real, loss_d_fake = os_exp[str(lam)]['losses']
    # loss_g, loss_d_real, loss_d_fake = gen_lr_exp[str(0.00018)]['losses']
    print(
        lam,
        f'G(z) = {loss_g[599]:.4f}', 
        f'D(x) = {loss_d_real[599]:.4f}', 
        f'D(G(z)) = {loss_d_fake[599]:.4f}'
    )
# %%
# Final Series Plot
for lam in [0.95,0.9,0.8,0.7,0.6]:
    for epoch in [599]:
        # plot_2d(mc2_exp_l['gen_samples'][epoch], plot_target=True, save_fig=f'2d_mc2_{epoch+1}.pdf')
        plot_2d(os_exp[str(lam)]['gen_samples'][epoch], plot_target=True)
# %%
def plot_metric(metric, exp_dicts, var_to_plot=[], fig_=plt.figure(), plt_config=(True,True,'',1)):
    l1, l2, l_title, plot_every = plt_config
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ["-","--"]
    ax_ = fig_.add_subplot(111)
    for j, exp_dict in enumerate(list(exp_dicts.values())) :
        if not var_to_plot: var_to_plot = [int(x) if int(float(x)) == float(x) else float(x) for x in list(exp_dict.keys())] 
        for i, exp_var in enumerate(var_to_plot):
            p_metric = get_metric(metric, exp_dict,exp_var)
            ax_.plot(np.arange(1, len(p_metric)+1, plot_every), 
                        p_metric[::plot_every], label=f'{exp_var}',color=color_cycle[i], linestyle=linestyle_cycle[j])
    
    lines = ax_.get_lines()
    # if l1: legend_1 = ax_.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]], title=l_title, ncol=len(lines)+1)
    if l1: legend_1 = ax_.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]], ncol=len(lines))
    if l2: legend_2 = ax_.legend(lines[::i+1], list(exp_dicts.keys()))
    if l1 and l2: ax_.add_artist(legend_1)
    
    ax_.set_xlabel('Epoch Number')
    ax_.set_ylabel(metric_names[metric] if metric in metric_names.keys() else metric)

    return ax_
#%%
exp_dicts4={
    # 'Varied Learning Rate': gen_lr_exp,
    'Varied Lambda': os_exp
    }
lam_to_plot=[0.95,0.9,0.8,0.7,0.6]
#%%
# Plot Loss
fig_x = plt.figure(figsize=(7.4, 2))
ax_x = plot_metric('d-loss',exp_dicts4, var_to_plot=lam_to_plot,fig_=fig_x, plt_config=(True, False, 'Lambda', 1))
# ax_x.set_ylim(0.685,0.695)
# ax_x.set_ylim(0.68,0.8)
fig_lr = ax_x.get_figure()
fig_lr.tight_layout()
# fig_lr.savefig('../report/figures/2d_gen_loss_ema.pdf', bbox_inches='tight', pad_inches=0)


#%%
# [EVAL Discriminator]
eval_dis=Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1))
eval_dis = eval_dis.to('cpu')
criterion = nn.MSELoss()
noise_fn=generate_noise
data_fn=sample_from_target_function
optim_d = optim.Adam(eval_dis.parameters(),lr=1e-3)
batch_size=64
for epoch in range(400):
        b_loss = []
        for batch in range(100):
            eval_dis.zero_grad()
            gen_latent_vec = noise_fn(batch_size // 2) # Half Noise Samples
            target = sample_from_target_function(batch_size // 2) # Half Real Samples
            samples = torch.from_numpy(np.concatenate((gen_latent_vec, target), axis=0).astype('float32')) # Combine Noise and Real
            labels = torch.from_numpy(np.concatenate((np.zeros((batch_size//2, 1)), np.ones((batch_size//2, 1))), axis=0).astype('float32')) # Create Labels
            conf = eval_dis(samples)
            loss = criterion(conf, labels)
            loss.backward()
            optim_d.step()
            ld_ = loss.item()
            b_loss.append(ld_)
        print(f"epoch={epoch}",
        f" D={np.mean(b_loss):.3f}")
# torch.save(eval_dis.state_dict(), './eval-dis.pt')

#%%
# Load eval-dis
eval_dis = Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1))
eval_dis.load_state_dict(torch.load('./eval-dis.pt'))
# eval_dis.eval()
#%%
GRID_RESOLUTION = 400
grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
grid[:, :, 1] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
flat_grid = grid.reshape((-1, 2))
torch_grid = torch.from_numpy(flat_grid.astype('float32'))
confd = eval_dis(torch_grid).detach().cpu().numpy()
confidences = confd.reshape((GRID_RESOLUTION, GRID_RESOLUTION))
plt.figure(figsize=(6,6))
plt.imshow(confidences, cmap='PiYG_r')
plt.xticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4), np.linspace(-1, 1, 5))
plt.yticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4), np.linspace(1, -1, 5))
plt.gca().set_aspect(1)
plt.tight_layout()
# plt.savefig('../report/figures/2d_dis_sup.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


# %%
# Eval Dis Accuaracy
target_ = sample_from_target_function(2000)
plot_2d(target_, plot_target=True)
confd = eval_dis(target_).detach().cpu().numpy()
print(
    f'Mean Conf = {np.mean(confd)*100:.2f}', 
    f'Std Conf = {np.std(confd):.4f}', 
    f'Accuracy = {((confd > 0.4).sum())/20}'
    )

# %%
concat_epoch = lambda gen_samples: np.concatenate((gen_samples[599], gen_samples[598]))
exp_dict = os_exp
for lam in [0.95,0.9,0.8,0.7,0.6]:
    # gen_sam= os_exp[str(lam)]['gen_samples'][599]
    lam_dict = exp_dict[str(lam)]
    gen_sam = concat_epoch(lam_dict['gen_samples'])
    plot_2d(gen_sam, plot_target=True)
    # plot_2d(os_exp[str(lam)]['gen_samples'][599], plot_target=True, save_fig=f'2d_os_lam_{lam*100:.0f}.pdf')
    confd = eval_dis(torch.tensor(gen_sam)).detach().cpu().numpy()
    print(
        f'Lambda = {lam}', 
        f'Mean Conf = {np.mean(confd)*100:.2f}', 
        f'Std Conf = {np.std(confd):.4f}', 
        f'Accuracy = {((confd > 0.4).sum())/confd.shape[0]*100:.2f}'
        )

#%%
exp_dict = gen_lr_exp
lam_dict = exp_dict[str(0.00018)]
gen_sam = concat_epoch(lam_dict['gen_samples'])
plot_2d(gen_sam, plot_target=True)
# plot_2d(os_exp[str(lam)]['gen_samples'][599], plot_target=True, save_fig=f'2d_os_lam_{lam*100:.0f}.pdf')
confd = eval_dis(torch.tensor(gen_sam)).detach().cpu().numpy()
print(
    f'Lambda = {lam}', 
    f'Mean Conf = {np.mean(confd)*100:.2f}', 
    f'Std Conf = {np.std(confd):.4f}', 
    f'Accuracy = {((confd > 0.4).sum())/confd.shape[0]*100:.2f}'
    )
#%%
# [Discriminator Eval Plot]
from matplotlib.colors import ListedColormap
concat_epoch = lambda gen_samples: np.concatenate((gen_samples[599], gen_samples[598]))
for lam in [0.95,0.9,0.8,0.7,0.6]:
    # plt.figure(figsize=(2.49,2.49))
    plt.figure(figsize=(8,8))
    test_gen = concat_epoch(os_exp[str(lam)]['gen_samples'])
    conf_test = eval_dis(torch.tensor(test_gen)).detach().cpu().numpy()
    input_points = generate_noise(4000)
    output_points = target_function(input_points)
    plt.scatter(output_points[:, 0], output_points[:, 1], c='blue',s=1, alpha=0.2, linewidth=1)
    cmap = ListedColormap('orange')
    cmap.set_under('red')
    plt.scatter(test_gen[:, 0], test_gen[:, 1], c=conf_test, cmap=cmap, vmin=0.5, vmax=1, s=1, alpha=0.8, linewidth=1)
    plt.gca().set_aspect(1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()    

#%%
for epoch in [49, 449, 559, 579, 599]:
    gen_sam = torch.tensor(mc2_exp_l['gen_samples'][epoch])
    plot_2d(mc2_exp_l['gen_samples'][epoch], plot_target=True)
    confd = eval_dis(gen_sam).detach().cpu().numpy()
    print(np.mean(confd))

# ------ Spherical --------

#%%
def sample_spherical(samples, hypersphere_dim=3):
    sphere_sample = np.random.randn(hypersphere_dim, samples)
    sphere_sample /= np.linalg.norm(sphere_sample, axis=0)
    return sphere_sample

#%%
gen_sample = os_exp['0.6']['gen_samples'][598]
latent_sample = sample_spherical(1000, hypersphere_dim=3).T[:,:2]
f, ax = coloured_plt(latent_sample,gen_sample)

#%%
def sample_spherical_noise(samples):
    return torch.from_numpy(sample_spherical(samples, 3).T[:,:2].astype('float32'))

# sample_spherical_noise(1000)

#%%
hyper_s_exp = load_pickle('hs_exp')
# hyper_s_exp = {}
exp_name_dict = hyper_s_exp
for exp_var in [1, 0.95, 0.6]:
    if str(exp_var) not in exp_name_dict.keys():
        print(f"Experiment Variable {exp_var} not found, running experiement")
        name_exp = V_GAN_smooth(
            generator=Generator(),
            discriminator=Discriminator(),
            noise_fn=sample_spherical_noise, # Sample from Spherical Noise
            data_fn=sample_from_target_function,
            plot_every = 50, 
            lr_d=1e-3, #1e-3
            lr_g=1.8e-4, #2e-4 , MC:1e-3
            betas_adam=(0.5, 0.999), # (0.9, 0.999) or (0.5, 0.999) for GAN
            epochs=600, 
            batches=100, 
            batch_size=128,
            gen_size=2000,
            device='cpu',
            one_sided = exp_var)
        exp_name_dict[str(exp_var)] = name_exp

save_pickle('hs_exp', exp_name_dict)

#%%
hs_exp = load_pickle('hs_exp')

#%%
concat_epoch = lambda gen_samples: np.concatenate((gen_samples[599], gen_samples[598]))
exp_dict = hs_exp
for lam in [1, 0.95, 0.6]:
    # gen_sam= os_exp[str(lam)]['gen_samples'][599]
    lam_dict = exp_dict[str(lam)]
    gen_sam = concat_epoch(lam_dict['gen_samples'])
    plot_2d(gen_sam, plot_target=True)
    # plot_2d(os_exp[str(lam)]['gen_samples'][599], plot_target=True, save_fig=f'2d_os_lam_{lam*100:.0f}.pdf')
    confd = eval_dis(torch.tensor(gen_sam)).detach().cpu().numpy()
    print(
        f'Lambda = {lam}', 
        f'Mean Conf = {np.mean(confd)*100:.2f}', 
        f'Std Conf = {np.std(confd):.4f}', 
        f'Accuracy = {((confd > 0.4).sum())/confd.shape[0]*100:.2f}'
        )

