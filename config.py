import experiment_buddy
import torch

algo = "ppo"  # no change
gail = False  # not important
gail_experts_dir = './gail_experts'  # not important
gail_batch_size = 128  # not important
gail_epoch = 5  # not important
lr = 2.5e-4
eps = 1e-5
alpha = 0.99  # for a2c not ppo
gamma = 0.99
use_gae = True
gae_lambda = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
seed = 1  # didnt change
cuda_deterministic = False
num_processes = 1
num_steps = 2500
custom_gym = "growspace"
ppo_epoch = 4
num_mini_batch = 32
clip_param = 0.1
log_interval = 10  # amount of times we save to wandb
save_interval = 100  # amount of times we save internal
eval_interval = None
num_env_steps = 1e6  # no change
env_name = "GrowSpaceEnv-Control-v0"  # "GrowSpaceSpotlight-Mnist4-v0"
log_dir = "/tmp/gym/"
save_dir = "../trained_models/"
use_proper_time_limits = False
recurrent_policy = False
use_linear_lr_decay = True
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
optimizer = "adam"
momentum = 0.95


stack_size = 4
seed = 123  # , help='Random seed')
T_max = int(50e6)
max_episode_length = int(108e3)

history_length = 4  # , metavar='T', help='Number of consecutive states processed')
hidden_size = 512  # , metavar='SIZE', help='Network hidden size')
noisy_std = 0.1  # , metavar='σ', help='Initial standard deviation of noisy linear layers')
atoms = 51  # , metavar='C', help='Discretised size of value distribution')
V_min = -10  # , metavar='V', help='Minimum of value distribution support')
V_max = 10  # , metavar='V', help='Maximum of value distribution support')
# model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
memory_capacity = int(1e6)  # , metavar='CAPACITY', help='Experience replay memory capacity')
replay_frequency = 4  # , metavar='k', help='Frequency of sampling from memory')
priority_exponent = 0.5  # , metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
priority_weight = 0.4  # , metavar='β', help='Initial prioritised experience replay importance sampling weight')
multi_step = 3  # , metavar='n', help='Number of steps for multi_step return')
discount = 0.99  # , metavar='γ', help='Discount factor')
target_update = int(32e3)  # , metavar='τ', help='Number of steps after which to update target network')
reward_clip = 1  # , metavar='VALUE', help='Reward clipping (0 to disable)')
lr = 0.0000625  # , metavar='η', help='Learning rate')
adam_eps = 1.5e-4  # , metavar='ε', help='Adam epsilon')
batch_size = 32  # , metavar='SIZE', help='Batch size')
# learn_start = int(80e3)  # , metavar='STEPS', help='Number of steps before starting training')
learn_start = 1_000
# evaluate # ='Evaluate only')
evaluation_interval = 1_000  # , metavar='STEPS', help='Number of training steps between evaluations')
evaluation_episodes = 10  # , metavar='N', help='Number of evaluation episodes to average over')
evaluation_size = 500  # , metavar='N', help='Number of transitions to use for validating Q')
log_interval = 25000  # , metavar='STEPS', help='Number of training steps between logging status')
evaluate = False
# render'Display screen (testing only)')
# save-dir', type=str, default='results_temp')

device = "cuda"

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "growspace"}
)
