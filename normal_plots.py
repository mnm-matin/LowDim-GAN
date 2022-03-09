
import statistics
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import torch

from utils import calc_bins, calc_p, calc_w, ema


def plot_his(test_gen, plot_normal = False, epoch='', save_fig='', labels=True, ylim=0.6):
    if plot_normal is True:
        plt.figure(figsize=(2,2), dpi=120) # 7.48 inches, 3.7
    
    if epoch is not '':
        plt.title("Histogram of Generated Distribution "+epoch)
    
    if labels is True:
        plt.xlabel("Value")
        plt.ylabel("Count")

    plt.hist(test_gen, density=True, bins=calc_bins(test_gen),rwidth=0.9)
    
    if plot_normal is True:
        # x = bins[:-1] + (bins[1] - bins[0])/2
        x = np.linspace(-3.5, 3.5, num=100)
        y = norm.pdf(x, 0, 1)
        plt.plot(x, y)

    plt.xlim([-3.25, 3.25])
    plt.ylim([0, ylim])

    plt.tight_layout()
    if save_fig is not '':
        plt.savefig('../report/figures/'+save_fig)
    plt.show()


#%%
metric_names = {
    'd-value':'d-value',
    'w-value':'w-value',
    'mean':'mean',
    'std':'standard deviation',
    'g-loss':'loss [G(z)]',
    'df-loss':'loss [D(G(z)]',
    'dr-loss': 'loss [D(x)]',
    'd-loss': 'loss 0.5*[D(G(z)+D(x)]',
    'd-loss-ema': 'loss (EMA)'
}
def get_metric(metric, exp_dict, LATENT_DIM):
    if metric == 'd-value': 
        p_metric = [calc_p(exp_dict[str(LATENT_DIM)]["gen_samples"][epoch]) for epoch in range(100)]
        p_metric = ema(p_metric, alpha=0.8)
    elif metric == 'w-value': 
        p_metric = [calc_w(exp_dict[str(LATENT_DIM)]["gen_samples"][epoch]) for epoch in range(100)]
    elif metric == 'mean': 
        p_metric, _ = exp_dict[str(LATENT_DIM)]['stats']
    elif metric == 'std': 
        _, p_metric = exp_dict[str(LATENT_DIM)]['stats']
        metric = 'standard deviation'
    elif metric == 'dr-loss':
        _, p_metric, _ = exp_dict[str(LATENT_DIM)]['losses']
        metric = 'D(x)'
    elif metric == 'df-loss':
        *_, p_metric = exp_dict[str(LATENT_DIM)]['losses']
        metric = 'D(G(z))'
    elif metric == 'dr-loss':
        _, p_metric, _ = exp_dict[str(LATENT_DIM)]['losses']
        metric = 'D(x)'
    elif metric == 'd-loss':
        p_metric = [statistics.mean(k) for k in zip(get_metric('dr-loss', exp_dict, LATENT_DIM),get_metric('df-loss', exp_dict, LATENT_DIM))]
        metric = 'Average Discriminator Loss'
    elif metric == 'd-loss-ema':
        p_metric = ema(get_metric('d-loss',exp_dict, LATENT_DIM),alpha=0.7)
    elif metric == 'g-loss':
        p_metric, *_ = exp_dict[str(LATENT_DIM)]['losses']
        metric = 'G(z)'
    else:
        print('must be one of d-value, w-value, mean, std, d-loss, g-loss')
    return p_metric

def plot_metric(metric, exp_dicts, dim_to_plot=[], fig_=plt.figure(), plt_config=(True,True)):
    l1, l2 = plt_config
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ["-","--"]

    ax_ = fig_.add_subplot(111)
    for j, exp_dict in enumerate(list(exp_dicts.values())) :
        if not dim_to_plot: dim_to_plot = [int(x) if int(float(x)) == float(x) else float(x) for x in list(exp_dict.keys())] 
        for i, LATENT_DIM in enumerate(dim_to_plot):
            p_metric = get_metric(metric, exp_dict,LATENT_DIM)
            ax_.plot(np.arange(1, len(p_metric)+1), 
                        p_metric, label=f'Latent Dimension {str(LATENT_DIM)}', color=color_cycle[i], linestyle=linestyle_cycle[j])
    
    lines = fig_.gca().get_lines()
    if l1: legend_1 = ax_.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]])
    if l2: legend_2 = ax_.legend(lines[::i+1], list(exp_dicts.keys()))
    if l1 and l2: fig_.gca().add_artist(legend_1)
    
    ax_.set_xlabel('Epoch Number')
    ax_.set_ylabel(metric_names[metric] if metric in metric_names.keys() else metric)

    return fig_











#%%
exp_dict = gan_lr_experiments
for epoch in [19,39,59,79,99,119,139,159,179,199]:
    for lr in [0.0001]:
        plot_his(exp_dict[str(lr)]["gen_samples"][epoch], epoch=f'epoch {epoch} for lr={lr}', plot_normal=True, labels = False)


#%%
import seaborn as sns
def plot_kde(fake_samples, save_fig=''):
    plt.figure(figsize=(2,2), dpi=120)
    #sns.kdeplot(test_samples.flatten(), c='blue', alpha=0.6, label='Real', ax=plt, shade=True)
    x_vals = np.linspace(-3, 3, 301)
    y_vals = stats.norm(0,1).pdf(x_vals)
    plt.plot(x_vals, y_vals, label='real')
    plt.fill_between(x_vals, np.zeros(len(x_vals)), y_vals, alpha=0.6)
    #sns.kdeplot(test_samples.flatten(), c='blue', alpha=0.6, label='Real', ax=plt, shade=True)
    try:
        sns.kdeplot(fake_samples.flatten(), alpha=0.6, label='GAN', ax=plt, shade=True)
    except Exception:
        pass
    plt.xlim(-3, 3)
    # plt.set_ylim(0, 0.82)
    # plt.legend(loc=1)
    # plt.xlabel('Sample Space')
    # plt.ylabel('Probability Density')
    plt.tight_layout()
    plt.savefig('../report/figures/'+save_fig) if save_fig is not '' else None
    plt.show()

# fake_samples = exp_dict[str(0.001)]["gen_samples"][199]
# plot_kde(fake_samples)
#%%
exp_dict = gan_lr_experiments
for lr in [0.01, 0.005, 0.001, 0.0005,0.0002, 0.0001]:
    for epoch in [50,199]:
        plot_kde(exp_dict[str(lr)]["gen_samples"][epoch])
#%%
exp_dict = gan_lr_experiments
for epoch in [50,199]:
    for lr in [0.01, 0.005, 0.001, 0.0005,0.0002, 0.0001]:
        plot_kde(exp_dict[str(lr)]["gen_samples"][epoch])

#%%
exp_dict = gan_lr_experiments
# for epoch in [19,39,59,79,99,119,139,159,179,199]:
for epoch in [199]:
    for lr in [0.0001]:
        plot_kde(exp_dict[str(lr)]["gen_samples"][epoch])

#%%
# get_metric
exp_dicts3={
    # 'Reduced Generator Complexity': gen_complexity_exp_loaded,
    # 'Reduced GAN Complexity': gan_complexity_exp_loaded,
    'Varied Learning Rate': gan_lr_experiments
    }
lr_to_plot=[0.01, 0.005, 0.001, 0.0005,0.0002, 0.0001]
lr_to_plot=[0.01, 0.005, 0.001, 0.0005,0.0002]
# lr_to_plot=[0.0002, 0.0001]
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('df-loss',exp_dicts3, dim_to_plot=lr_to_plot,fig_=fig_x)
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('d-loss',exp_dicts3, dim_to_plot=lr_to_plot,fig_=fig_x)
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('d-loss-ema',exp_dicts3, dim_to_plot=lr_to_plot,fig_=fig_x)
#%%
fig_x = plt.figure(figsize=(8, 5))
plot_metric('g-loss',exp_dicts3, dim_to_plot=lr_to_plot,fig_=fig_x)


#%%
def plot_metric(metric, exp_dicts, dim_to_plot=[], fig_=plt.figure(), plt_config=(True,True)):
    l1, l2 = plt_config
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyle_cycle = ["-","--"]

    ax_ = fig_.add_subplot(111)
    for j, exp_dict in enumerate(list(exp_dicts.values())) :
        if not dim_to_plot: dim_to_plot = [int(x) if int(float(x)) == float(x) else float(x) for x in list(exp_dict.keys())] 
        for i, LATENT_DIM in enumerate(dim_to_plot):
            p_metric = get_metric(metric, exp_dict,LATENT_DIM)[:-25]
            ax_.plot(np.arange(1, len(p_metric)+1), 
                        p_metric, label=f'{LATENT_DIM}', color=color_cycle[i], linestyle=linestyle_cycle[j])
    ax_.set_ylim(0.4,0.8)
    lines = fig_.gca().get_lines()
    if l1: legend_1 = ax_.legend(lines[:i+1], [l.get_label() for l in lines[:i+1]], title="Learning Rate", loc=5)
    if l2: legend_2 = ax_.legend(lines[::i+1], list(exp_dicts.keys()))
    if l1 and l2: fig_.gca().add_artist(legend_1)
    
    ax_.set_xlabel('Epoch Number')
    ax_.set_ylabel(metric_names[metric] if metric in metric_names.keys() else metric)

    return ax_
#%%
p_dlr_ema = plt.figure(figsize=(7.4, 2))
plot_metric('d-loss-ema',exp_dicts3, dim_to_plot=[0.01, 0.005, 0.001, 0.0005,0.0002],fig_=p_dlr_ema, plt_config=(True,False))
p_dlr_ema.tight_layout()
p_dlr_ema.savefig('../report/figures/1d_exp3_emal.pdf')

#%%
exp_dict = gan_lr_experiments
for epoch in [199]:
    for lr in [0.01, 0.005, 0.001, 0.0005,0.0002]:
        plot_kde(exp_dict[str(lr)]["gen_samples"][epoch],save_fig=f'1d_exp3_kde_lr_{lr}.pdf')

#%%
noise_fn = lambda x: torch.rand((x, 1), device='cpu')
data_fn = lambda x: torch.randn((x, 1), device='cpu')

# %%
test_size = 2000

test_latent_vec = noise_fn(test_size)
test_noise = test_latent_vec.cpu().numpy().flatten()
test_noise = np.sort(test_noise)

test_real_samples = data_fn(test_size).cpu().numpy().flatten()
test_real_samples = np.sort(test_real_samples)
# %%
# PLOT 1D NOISE
plt.title("Noise")
plt.xlim([-2.5, 2.5])
plt.ylabel('Density')
plt.hist(test_noise, density=True, bins=calc_bins(test_noise), rwidth=0.9)
plt.tight_layout()
plt.savefig('../report/figures/1d_noise.pdf')

# %%
# PLOT REAL SAMPLES
plt.title("Real Samples")
plt.ylabel('Density')
plt.xlim([-3.5, 3.5])
plt.ylim([0, 0.8])
plt.hist(test_real_samples, density=True, bins=calc_bins(test_real_samples), color='red', rwidth=0.9)
plt.tight_layout()
plt.savefig('../report/figures/real_samples.pdf')


# %%
# PLOT 2D NOISE
noise_fn_2 = lambda x: torch.rand((x, 2), device='cpu')
test_size = 1500
test_latent_vec = noise_fn_2(test_size)
test_noise = test_latent_vec.cpu().numpy()
x_points = [x[0] for x in test_noise]
y_points = [x[1] for x in test_noise]
plt.figure(figsize=(4,2.5), dpi=100)
plt.xlim([-3.25, 3.25])
plt.ylim([-0.1, 1.05])
plt.xlabel('X')
plt.ylabel('Y')
noise_2 = plt.scatter(x_points,y_points, s=1)
test_real_samples = data_fn(test_size).cpu().numpy().flatten()
real_1 = plt.scatter(test_real_samples,[0]*len(test_real_samples), s=1.5, alpha=0.05)
plt.legend((noise_2, real_1),('Noise','Data'), scatterpoints=100)
plt.tight_layout()
# plt.show()
plt.savefig('../report/figures/2d_noise.pdf')




#%%
# [1, 0]: Confident real input
grid_latent = np.linspace(-1, 1, 2002)[1:-1].reshape((-1, 1))
print(len(grid_latent.flatten()))
true_mappings = test_real_samples
GAN_mapping = np.sort(gan_complexity_exp_loaded[str(1)]["gen_samples"][99])
plt.scatter(grid_latent.flatten(), true_mappings, 
            edgecolor='blue', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='Real Mapping')
plt.scatter(grid_latent.flatten(), GAN_mapping, 
            edgecolor='red', facecolor='None', s=5, alpha=1, 
            linewidth=1, label='GAN Mapping')
plt.legend(loc=8)
plt.xlim(-1, 1)
plt.ylim(-3, 3)
plt.xlabel('Latent Space')
plt.ylabel('Sample Space')


# %%
# test_gen = generator(test_latent_vec).detach().cpu().numpy().flatten()
# test_gen = np.sort(test_gen)
plt.cla()
plt.scatter(test_real_samples, np.linspace(0, 1, len(test_real_samples)), c='k', s=2)
plt.scatter(test_gen, np.linspace(0, 1, len(test_gen)), c='r', s=2)


#%%
# PLOT NETWORK ARCHITECTURE
from torchviz import make_dot
generator=Normal_Generator(2)
noise_fn = lambda x: torch.rand((x, 1), device='cpu')
latent_vec = noise_fn(2).detach()
gen_out = generator(latent_vec)
print(dict(generator.named_parameters()).keys())
make_dot(gen_out, params=dict(generator.named_parameters()), show_saved=False)
# from graphviz import Source; model_arch = make_dot(...); Source(model_arch).render(filepath);