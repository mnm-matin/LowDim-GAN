from decimal import Decimal
import pickle
import statistics

import torch
from torch import nn
import torch.optim as optim

from normal_models import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
from normal_plots import get_metric, plot_his, plot_metric
from utils import calc_bins, calc_p, calc_w, count_parameters, ema, load_pickle, save_pickle


def Normal_GAN(
    generator=Normal_Generator(1),
    discriminator=Normal_Discriminator(1),
    noise_fn=lambda x: torch.rand((x, 1), device='cpu'),
    data_fn=lambda x: torch.randn((x, 1), device='cpu'),
    plot_every = 0, 
    lr_d=1e-3,
    lr_g=2e-4, 
    epochs=100, 
    batches=100, 
    batch_size=32,
    device='cpu'):

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion = nn.BCELoss()
    optim_d = optim.Adam(discriminator.parameters(),
                                lr=lr_d, betas=(0.5, 0.999))
    optim_g = optim.Adam(generator.parameters(),
                                lr=lr_g, betas=(0.5, 0.999))
    target_ones = torch.ones((batch_size, 1)).to(device)
    target_zeros = torch.zeros((batch_size, 1)).to(device)
    loss_g, loss_d_real, loss_d_fake = [], [], []
    gen_samples = []
    fake_mean, fake_std = [], []
    start = time()
    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        fake_mean_running, fake_std_running = 0, 0
        for batch in range(batches):

            # Train the generator one step
            generator.zero_grad()
            latent_vec = noise_fn(batch_size)
            generated = generator(latent_vec)
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
            latent_vec = noise_fn(batch_size)
            fake_samples = generator(latent_vec).detach()
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

            fake_mean_running += fake_samples.mean()
            fake_std_running += fake_samples.std()

        loss_g.append(loss_g_running / batches)
        loss_d_real.append(loss_d_real_running / batches)
        loss_d_fake.append(loss_d_fake_running / batches)
        fake_mean.append(fake_mean_running / batches)
        fake_std.append(fake_std_running / batches)

        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
                f" G={loss_g[-1]:.3f},"
                f" D(x)={loss_d_real[-1]:.3f},"
                f" D(G(z))={loss_d_fake[-1]:.3f}",
                f" Mean(G(z))={fake_mean[-1]:.3f}",
                f" Std(G(z))={fake_std[-1]:.3f}")

        def generate_sample(test_size=2000):
            test_latent_vec = noise_fn(test_size)
            test_gen = generator(test_latent_vec).detach().cpu().numpy().flatten()
            return test_gen, test_latent_vec

        test_gen, test_latent_vec = generate_sample(2000)
        gen_samples.append(test_gen)

        if plot_every != 0 and (epoch+1) % plot_every == 0:
            plot_his(test_gen, plot_normal=True, epoch="at Epoch "+str(epoch+1))
    
    model_dict ={
        "gen_samples": gen_samples,
        "losses": (loss_g, loss_d_real, loss_d_fake),
        "stats": (fake_mean, fake_std)
    }

    return model_dict

#%%
original_model = Normal_GAN(plot_every=10)

#%%
latent_experiments = {}
for LATENT_DIM in [1,2,4,10,64]:
    latent_exp = Normal_GAN(
        generator=Normal_Generator(LATENT_DIM),
        discriminator=V_Discriminator(1, [64, 32, 1]),
        noise_fn=lambda x: torch.rand((x, LATENT_DIM), device='cpu'),
        data_fn=lambda x: torch.randn((x, 1), device='cpu'),
        plot_every = 10, 
        lr_d=1e-3,
        lr_g=2e-4, 
        epochs=100, 
        batches=100, 
        batch_size=32,
        device='cpu')
    latent_experiments[str(LATENT_DIM)] = latent_exp

save_pickle('latent_exp',latent_experiments)
#%%
# Load Latent Experiment
with open('latent_exp.pickle', 'rb') as handle:
    latent_exp_loaded = pickle.load(handle)
#%%
# TEST PLOT
for epoch in [99]:
    for LATENT_DIM in [1,2,4,10,64]:
        plot_his(latent_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch], plot_normal=True, labels = False)

#%%
# VARYING EPOCHS
for epoch in [19,39,59,79,99]:
    for LATENT_DIM in [1]:
        plot_his(latent_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch],plot_normal=True,labels=False, epoch='none', save_fig=f'1d_gen_{LATENT_DIM}_{epoch+1}.pdf')
#%%
# VARYING LATENT_DIM
for epoch in [99]:
    for LATENT_DIM in [1,2,4,10,64]:
        plot_his(latent_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch],plot_normal=True,labels=False, epoch='none', save_fig=f'1d_gen_{LATENT_DIM}_{epoch+1}.pdf')

#%% [markdown]
#### Count Paramaters
for LATENT_DIM in [1,2,4,16,32,64]:
    print(f'LATENT_DIM: {LATENT_DIM} Paramaters: {count_parameters(Normal_Generator(LATENT_DIM))}')
print('-')
for LATENT_DIM in [1,2,4,16,32,64]:
    print(f'LATENT_DIM: {LATENT_DIM} Paramaters: {count_parameters(V_Generator(LATENT_DIM, [32, 1]))}')

#%% [markdown]
# Reduce Generator Complexity
gen_complexity_experiments = load_pickle('gen_complexity_exp')
for LATENT_DIM in [1,2,4,8,10,16,32,64]:
    if str(LATENT_DIM) not in gen_complexity_experiments.keys():
        print(f"LATENT DIM {LATENT_DIM} not found, running experiement")
        gen_complexity_exp = Normal_GAN(
            generator=V_Generator(LATENT_DIM, [32, 1]),
            discriminator=V_Discriminator(1, [64, 32, 1]),
            noise_fn=lambda x: torch.rand((x, LATENT_DIM), device='cpu'),
            data_fn=lambda x: torch.randn((x, 1), device='cpu'),
            plot_every = 10, 
            lr_d=1e-3,
            lr_g=2e-4, 
            epochs=100, 
            batches=100, 
            batch_size=32,
            device='cpu')
        gen_complexity_experiments[str(LATENT_DIM)] = gen_complexity_exp

save_pickle('gen_complexity_exp', gen_complexity_experiments)

#%%
# Load Generator Complexity Experiment
gen_complexity_exp_loaded = load_pickle('gen_complexity_exp')

#%%
# TEST PLOT
for epoch in [99]:
    for LATENT_DIM in [1,2,4,10,16,32,64]:
        plot_his(gen_complexity_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch], plot_normal=True, epoch=f'epoch {epoch} for k={LATENT_DIM}', labels = False)

#%%
# TEST PLOT
for epoch in [19,39,59,79,99]:
    for LATENT_DIM in [2]:
        plot_his(gen_complexity_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch], plot_normal=True, epoch=f'epoch {epoch} for k={LATENT_DIM}', labels = False)

# PLOT W_stat and P Stat
#%%
def plot_w_stats(exp_dicts, dim_to_plot=[], fig_=plt.figure(figsize=(8, 4))):
    # if not dim_to_plot: dim_to_plot = [int(x) for x in list(list(exp_dicts.values(keys())]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ["-","--"]
    ax_1 = fig_.add_subplot(111)
    for j, exp_dict in enumerate(list(exp_dicts.values())) :
        for i, LATENT_DIM in enumerate(dim_to_plot) :
            w_list = [calc_w(exp_dict[str(LATENT_DIM)]["gen_samples"][epoch]) for epoch in range(100)]
            ax_1.plot(np.arange(1, len(w_list)+1), 
                        w_list, label=f'Latent Dimension {str(LATENT_DIM)}', color=color_cycle[i], linestyle=linestyle_cycle[j])

    lines = plt.gca().get_lines()
    legend_1 = ax_1.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]])
    # legend_2 = ax_1.legend(lines[::i+1], list(exp_dicts.keys()), loc=3)
    # plt.gca().add_artist(legend_1)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('W-value')
    # ax_1.set_ylim(-0.00001,0.002)

    return fig_, ax_1

def plot_p_stats(exp_dicts, dim_to_plot=[], fig_=plt.figure(figsize=(8, 4))):
    # if not dim_to_plot: dim_to_plot = [int(x) for x in list(list(exp_dicts.values(keys())]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ["-","--"]
    ax_1 = fig_.add_subplot(111)
    for j, exp_dict in enumerate(list(exp_dicts.values())) :
        for i, LATENT_DIM in enumerate(dim_to_plot) :
            p_list = [calc_p(exp_dict[str(LATENT_DIM)]["gen_samples"][epoch]) for epoch in range(100)]
            ax_1.plot(np.arange(1, len(p_list)+1), 
                        ema(p_list,alpha=0.8), label=f'Latent Dimension {str(LATENT_DIM)}', color=color_cycle[i], linestyle=linestyle_cycle[j])

    lines = plt.gca().get_lines()
    # legend_1 = ax_1.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]], bbox_to_anchor=(0.05,-0.001), loc="upper left", bbox_transform=fig_.transFigure, ncol=3)
    legend_2 = ax_1.legend(lines[::i+1], list(exp_dicts.keys()), loc=2)
    # plt.gca().add_artist(legend_1)
    ax_1.set_xlabel('Epoch number')
    ax_1.set_ylabel('D-value')
    ax_1.set_ylim(0,600)

    return fig_, ax_1

#%%
w_f = plt.figure(figsize=(3.7, 5))
w_f, ax_ = plot_w_stats(
    {'Original Experiment': latent_exp_loaded,
    'Reduced Generator Complexity': gen_complexity_exp_loaded},
    dim_to_plot=[1,2,64], 
    fig_=w_f)
w_f.tight_layout()
# w_f.savefig('../report/figures/1d_1_2_64_w_stat.pdf')
#%%
p_f = plt.figure(figsize=(3.7, 5))
p_f, ax_ = plot_p_stats(
    {'Original Experiment': latent_exp_loaded,
    'Reduced Generator Complexity': gen_complexity_exp_loaded},
    dim_to_plot=[1,2,64],
    fig_=p_f)
p_f.tight_layout()
# p_f.savefig('../report/figures/1d_1_2_64_p_stat.pdf')

#%%


#%%
# Reduce Generator & Discriminator Complexity
gan_complexity_experiments = load_pickle('gan_complexity_exp')
for LATENT_DIM in [1, 2, 4, 8, 10, 16, 32, 64]:
    if str(LATENT_DIM) not in gan_complexity_experiments.keys():
        print(f"LATENT DIM {LATENT_DIM} not found, running experiement")
        gan_complexity_exp = Normal_GAN(
            generator=V_Generator(LATENT_DIM, [32, 1]),
            discriminator=V_Discriminator(1, [32, 1]),
            noise_fn=lambda x: torch.rand((x, LATENT_DIM), device='cpu'),
            data_fn=lambda x: torch.randn((x, 1), device='cpu'),
            plot_every = 10, 
            lr_d=1e-3,
            lr_g=2e-4, 
            epochs=100, 
            batches=100, 
            batch_size=32,
            device='cpu')
        gan_complexity_experiments[str(LATENT_DIM)] = gan_complexity_exp
save_pickle('gan_complexity_exp', gan_complexity_experiments)

#%%
# Load Complexity Experiment
gan_complexity_exp_loaded = load_pickle('gan_complexity_exp')
#%%
# TEST PLOT
for epoch in [99]:
    for LATENT_DIM in [1,2,8]:
        plot_his(gan_complexity_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch], epoch=f'epoch {epoch} for k={LATENT_DIM}', plot_normal=True, labels = False)

#%%
# TEST PLOT
for epoch in [19,39,59,79,99]:
    for LATENT_DIM in [2]:
        plot_his(gan_complexity_exp_loaded[str(LATENT_DIM)]["gen_samples"][epoch], epoch=f'epoch {epoch} for k={LATENT_DIM}', plot_normal=True, labels = False)

#%%
w_f = plt.figure(figsize=(3.7, 5))
w_f, ax_ = plot_w_stats(
    {'Reduced Generator Complexity': gen_complexity_exp_loaded,
    'Reduced GAN Complexity': gan_complexity_exp_loaded},
    dim_to_plot=[1,2,64], 
    fig_=w_f)
w_f.tight_layout()
# w_f.savefig('../report/figures/1d_1_2_64_w_stat.pdf')
#%%
p_f = plt.figure(figsize=(3.7, 5))
p_f, ax_ = plot_p_stats(
    {'Reduced Generator Complexity': gen_complexity_exp_loaded,
    'Reduced GAN Complexity': gan_complexity_exp_loaded},
    dim_to_plot=[1,2,64],
    fig_=p_f)
p_f.tight_layout()
# p_f.savefig('../report/figures/1d_1_2_64_p_stat.pdf')



#%%
exp_dicts2={
    'Reduced Generator Complexity': gen_complexity_exp_loaded,
    'Reduced GAN Complexity': gan_complexity_exp_loaded
    }

fig_s = plt.figure(figsize=(3.7, 5))
_ = plot_metric('d-value',exp_dicts2, dim_to_plot=[1,2,4],fig_=fig_s)
fig_s = plt.figure(figsize=(3.7, 5))
_ = plot_metric('w-value',exp_dicts2, dim_to_plot=[1,2,4],fig_=fig_s)

#%%
fig_s = plt.figure(figsize=(3.7, 5))
_ = plot_metric('mean',exp_dicts2, dim_to_plot=[1,2,64],fig_=fig_s)
fig_s = plt.figure(figsize=(3.7, 5))
_ = plot_metric('std',exp_dicts2, dim_to_plot=[1,2,64],fig_=fig_s)

#%%
exp2_d_loss = plt.figure(figsize=(7.4, 2))
exp2_d_loss = plot_metric('d-loss',exp_dicts2, dim_to_plot=[1,2,8],fig_=exp2_d_loss, plt_config=(True,False))
exp2_d_loss.tight_layout()
# exp2_d_loss.savefig('../report/figures/1d_exp2_d_loss.pdf')
#%%
exp2_g_loss = plt.figure(figsize=(7.4, 2))
exp2_g_loss = plot_metric('g-loss',exp_dicts2, dim_to_plot=[1,2,8],fig_=exp2_g_loss, plt_config=(False, True))
exp2_g_loss.tight_layout()
# exp2_g_loss.savefig('../report/figures/1d_exp2_g_loss.pdf')

#%%
exp2_dr_loss = plt.figure(figsize=(7.4, 2))
exp2_dr_loss = plot_metric('dr-loss',exp_dicts2, dim_to_plot=[1,2,8],fig_=exp2_dr_loss, plt_config=(True,False))

#%%
exp2_dr_loss = plt.figure(figsize=(7.4, 2))
exp2_dr_loss = plot_metric('df-loss',exp_dicts2, dim_to_plot=[1,2,8],fig_=exp2_dr_loss, plt_config=(True,False))


#%%
get_metric('d-loss', gan_complexity_exp_loaded,8)[0]
#%%
get_metric('dr-loss', gan_complexity_exp_loaded,8)[0]
#%%
get_metric('df-loss', gan_complexity_exp_loaded,8)[0]


#%%
for metric in ['g-loss', 'd-loss', 'd-value', 'w-value']:
    for epoch in [79]:
        for LATENT_DIM in [1,2,8]:
                print(f'{metric} for {LATENT_DIM} at Epoch {epoch} is {get_metric(metric=metric, exp_dict = gan_complexity_exp_loaded, LATENT_DIM=LATENT_DIM)[epoch]}')


#%%
gan_lr_experiments = load_pickle('gan_lr_exp')
exp_name_dict = gan_lr_experiments
LATENT_DIM = 2
# 0.01, 0.005, *0.001, 0.0005,*0.0002, 0.0001 
for lr in [0.01, 0.005, 0.001, 0.0005,0.0002, 0.0001]:
    if str(lr) not in exp_name_dict.keys():
        print(f"Learning Rate {lr} not found, running experiement")
        name_exp = Normal_GAN(
            generator=V_Generator(LATENT_DIM, [32, 1]),
            discriminator=V_Discriminator(1, [32, 1]),
            noise_fn=lambda x: torch.rand((x, LATENT_DIM), device='cpu'),
            data_fn=lambda x: torch.randn((x, 1), device='cpu'),
            plot_every = 10, 
            lr_d=lr,
            lr_g=2e-4, 
            epochs=200, 
            batches=100, 
            batch_size=32,
            device='cpu')
        exp_name_dict[str(lr)] = name_exp
save_pickle('gan_lr_exp', exp_name_dict)

#%%
gan_lr_experiments = load_pickle('gan_lr_exp')


