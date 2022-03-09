from matplotlib import cm, pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from rose_gan import generate_noise, sample_from_target_function, sample_noise, sample_spherical, target_function
from rose_models import Discriminator, Generator


def plot_2d(test_gen, plot_target = False, epoch='', save_fig=''):  
    if epoch is not '':
        plt.figure(figsize=(4,4))
        plt.title("Scatter Plot of Generated Distribution "+epoch) #7.48 inches
    else:
        plt.figure(figsize=(1.49,1.49))
    
    if plot_target is True:
        input_points = generate_noise(2000)
        input_points[:,1]=0
        output_points = target_function(input_points)
        plt.scatter(output_points[:, 0], output_points[:, 1], edgecolor='blue', facecolor='None', s=5, alpha=0.1, linewidth=1)

    plt.scatter(test_gen[:, 0], test_gen[:, 1], edgecolor='orange', facecolor='None', s=5, alpha=0.9, linewidth=1)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.gca().set_aspect(1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_fig is not '':
        plt.savefig('../report/figures/'+save_fig, bbox_inches='tight', pad_inches=0)
    plt.show()


#%%
# [Discriminator Eval Plot]
concat_epoch = lambda gen_samples: np.concatenate((gen_samples[599], gen_samples[598]))
exp_dict = hs_exp
for lam in [1, 0.95, 0.6]:
    plt.figure(figsize=(2.49,2.49))
    # plt.figure(figsize=(3.7,3.7))
    # plt.figure(figsize=(8,8))
    test_gen = concat_epoch(exp_dict[str(lam)]['gen_samples'])
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
    # plt.savefig(f'../report/figures/2d_hs_lam_{lam*100:.0f}.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


#%%
# Colour Coded Plot, non-regularized, 
# 1 - Non-regularized from hypersphere
# 2 - Heavily-Regularized from uniform distribution
input_points = hs_exp['1']['latent_samples'][599]
output_points = hs_exp['1']['gen_samples'][599]
N = input_points.shape[0]
c = cm.rainbow(np.linspace(0, 1, N))
idx   = np.argsort(input_points[:, 0])
input_points_s = np.array(input_points)[idx]
output_points_s = np.array(output_points)[idx]

f, ax = plt.subplots(1, 2, figsize=(7.4, 3.7))
ax[0].scatter(input_points_s[:, 0], input_points_s[:, 1],
                edgecolor=c, facecolor='None', s=5, alpha=1, linewidth=1)
ax[1].scatter(output_points_s[:, 0], output_points_s[:, 1],
                edgecolor=c, facecolor='None', s=5, alpha=1, linewidth=1)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
ax[0].set_aspect(1)
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_aspect(1)
plt.tight_layout()
# plt.savefig(f'../report/figures/2d_coloured_hs_100.pdf', bbox_inches='tight', pad_inches=0)

# plt.show()
# return f, ax
#%%
input_points = os_exp['0.6']['latent_samples'][599]
output_points = os_exp['0.6']['gen_samples'][599]
N = input_points.shape[0]
c = cm.rainbow(np.linspace(0, 1, N))
idx   = np.argsort(input_points[:, 0])
input_points_s = np.array(input_points)[idx]
output_points_s = np.array(output_points)[idx]

f, ax = plt.subplots(1, 2, figsize=(7.4, 3.7))
ax[0].scatter(input_points_s[:, 0], input_points_s[:, 1],
                edgecolor=c, facecolor='None', s=5, alpha=1, linewidth=1)
ax[1].scatter(output_points_s[:, 0], output_points_s[:, 1],
                edgecolor=c, facecolor='None', s=5, alpha=1, linewidth=1)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
ax[0].set_aspect(1)
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_aspect(1)
plt.tight_layout()
# plt.savefig(f'../report/figures/2d_coloured_os_60.pdf', bbox_inches='tight', pad_inches=0)

# %%
# Defining the Problem [Same Axis]
f, ax = plt.subplots(1, 2, figsize=(12,6))
input_points = generate_noise(5000)
output_points = target_function(input_points)
ax[0].scatter(input_points[:, 0], input_points[:, 1], edgecolor='slategrey', facecolor='None', s=5, alpha=1, linewidth=1)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
ax[0].set_aspect(1)
ax[1].scatter(output_points[:, 0], output_points[:, 1], edgecolor='darkmagenta', facecolor='None', s=5, alpha=1, linewidth=1)
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_aspect(1)
plt.tight_layout()
# %%
# Defining the Problem 1 [NOISE]
input_points = generate_noise(3500)
plt.figure(figsize=(3.7,3.7))
plt.scatter(input_points[:, 0], input_points[:, 1], edgecolor='slategrey', facecolor='None', s=5, alpha=1, linewidth=1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect(1)
plt.locator_params(axis='y', nbins=2)
plt.locator_params(axis='x', nbins=2)
plt.tight_layout()
# plt.savefig('../report/figures/2d_noise_2.pdf')
plt.show()
# %%
# Defining the Problem 2 [PETAL]
output_points = target_function(input_points)
plt.figure(figsize=(3.7,3.7))
plt.scatter(output_points[:, 0], output_points[:, 1], edgecolor='darkmagenta', facecolor='None', s=5, alpha=1, linewidth=1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect(1)
plt.locator_params(axis='y', nbins=2)
plt.locator_params(axis='x', nbins=2)
plt.tight_layout()
# plt.savefig('../report/figures/2d_target_2.pdf')
plt.show()
#%%
# Contour Plots
sqrt_samples = 75
noise = generate_noise(sqrt_samples**2)
x = noise[:,0]
y = noise[:,1]
# x = np.random.uniform(-1, 1, size=10000)
# y = np.random.uniform(-1, 1, size=10000)
z = target_function(np.stack((x, y), axis=1))
x_=z[:,0]
y_=z[:,1]
xyx_ = np.array([x, y, x_]).T
xyy_ = np.array([x, y, y_]).T
# Contour Plot 1 [X-AXIS]
x, y, x_ = xyx_[xyx_[:, 0].argsort()].T
X, Y, X_ = (u.reshape(sqrt_samples, sqrt_samples) for u in (x, y, x_))
plt.figure(figsize=(3.7,2))
plt.contour(X, Y, X_)
plt.tight_layout()
# plt.savefig('../report/figures/2d_contour_x.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
#%%
# Contour Plot 2 [Y-AXIS]
x, y, y_ = xyy_[xyy_[:, 0].argsort()].T
X, Y, Y_ = (u.reshape(sqrt_samples, sqrt_samples) for u in (x, y, y_))
plt.figure(figsize=(3.7,2))
plt.contour(X, Y, Y_)
plt.tight_layout()
# plt.savefig('../report/figures/2d_contour_y.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

#%%
# GENERATOR Complexity Test
generator=Generator(layer_size=[2,512,512,512,2], layer_activation=nn.LeakyReLU(0.1))
generator = generator.to('cpu')
# criterion = nn.BCELoss()
criterion = nn.MSELoss()
noise_fn=sample_noise
data_fn=lambda Z: torch.from_numpy(target_function(Z).astype('float32'))
optim_g = optim.Adam(generator.parameters(),lr=2e-4)
batch_size=32
for epoch in range(40):
        for batch in range(100):
            generator.zero_grad()
            gen_latent_vec = noise_fn(batch_size)
            generated = generator(gen_latent_vec)
            target = data_fn(gen_latent_vec)
            loss = criterion(generated, target)
            loss.backward()
            optim_g.step()
            lg_ = loss.item()
            print(f"epoch={epoch}",
            f" G={lg_:.3f}")

gen_latent_vec = noise_fn(1000)
gen_sample = generator(gen_latent_vec).detach().cpu().numpy()
plot_2d(gen_sample,plot_target=True)
f, ax = coloured_plt(gen_latent_vec,gen_sample)
f.savefig('../report/figures/2d_gen_sup.pdf', bbox_inches='tight', pad_inches=0)

#%%
# DISCRIMINATOR Complexity Test
discriminator=Discriminator(layer_size=[2,512,512,512,1], layer_activation=nn.LeakyReLU(0.1))
discriminator = discriminator.to('cpu')
# criterion = nn.BCELoss()
criterion = nn.MSELoss()
noise_fn=generate_noise
data_fn=sample_from_target_function

optim_d = optim.Adam(discriminator.parameters(),lr=1e-3)
batch_size=32
for epoch in range(200):
        for batch in range(100):
            discriminator.zero_grad()
            gen_latent_vec = noise_fn(batch_size // 2) # Half Noise Samples
            target = sample_from_target_function(batch_size // 2) # Half Real Samples
            samples = np.concatenate((gen_latent_vec, target), axis=0) # Combine Noise and Real
            labels = np.concatenate((np.zeros((batch_size//2, 1)), np.ones((batch_size//2, 1))), axis=0) # Create Labels
            samples = torch.from_numpy(samples.astype('float32'))
            labels = torch.from_numpy(labels.astype('float32'))
            # print(samples)
            # print(labels)
            conf = discriminator(samples)
            loss = criterion(conf, labels)
            loss.backward()
            optim_d.step()
            ld_ = loss.item()
            print(f"epoch={epoch}",
            f" D={ld_:.3f}")

GRID_RESOLUTION = 400
grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
grid[:, :, 1] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
flat_grid = grid.reshape((-1, 2))
torch_grid = torch.from_numpy(flat_grid.astype('float32'))
confd = discriminator(torch_grid).detach().cpu().numpy()
confidences = confd.reshape((GRID_RESOLUTION, GRID_RESOLUTION))
plt.figure(figsize=(6,6))
plt.imshow(confidences, cmap='PiYG_r')
plt.xticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4), np.linspace(-1, 1, 5))
plt.yticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4), np.linspace(1, -1, 5))
plt.gca().set_aspect(1)
plt.tight_layout()
# plt.savefig('../report/figures/2d_dis_sup.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
# plt.close()
#%%
# [Hypersphere 3D Plot]
from mpl_toolkits.mplot3d import axes3d

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

xi, yi, zi = sample_spherical(1000)
plt.figure(figsize=(3.7,3.7))
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
ax.scatter(xi, yi, zi, s=5, c='r', zorder=10)
ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
plt.locator_params(axis='y', nbins=2)
plt.locator_params(axis='x', nbins=2)
plt.locator_params(axis='z', nbins=2)
fig.tight_layout()
# fig.savefig('../report/figures/2d_hyper_s.pdf', bbox_inches='tight', pad_inches=0)


# %%
# [Hypershphere 2d Projection]
input_points = sample_spherical(3000, ndim=3).T[:,:2]
plt.figure(figsize=(3.7,3.7))
plt.scatter(input_points[:, 0], input_points[:, 1], edgecolor='slategrey', facecolor='None', s=5, alpha=1, linewidth=1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect(1)
plt.locator_params(axis='y', nbins=2)
plt.locator_params(axis='x', nbins=2)
plt.tight_layout()
# plt.savefig('../report/figures/2d_hyper_n.pdf',bbox_inches='tight', pad_inches=0)
plt.show()


#%%
# Contour Plot [Sphere]
sqrt_samples = 75
noise = sample_spherical(sqrt_samples**2, ndim=3).T
x = noise[:,0]
y = noise[:,1]
z = noise[:,2]
xyz = np.array([x, y, z]).T
x, y, z = xyz[xyz[:, 0].argsort()].T
X, Y, Z = (u.reshape(sqrt_samples, sqrt_samples) for u in (x, y, z))
plt.figure(figsize=(3.7,3.7))
plt.contour(X, Y, Z)
plt.locator_params(axis='y', nbins=4)
plt.locator_params(axis='x', nbins=4)
plt.gca().set_aspect(1)
plt.tight_layout()
# plt.savefig('../report/figures/2d_contour_x.pdf', bbox_inches='tight', pad_inches=0)
plt.show()