import argparse
import torch

from utils import make_envs, take_action, init_obj
from feudalnet import FeudalNetwork, feudal_loss
from storage import Storage
from logger import Logger


parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='MiniGrid-FourRooms-v0',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=16,
                    help='number of parallel environments to run')
parser.add_argument('--num-steps', type=int, default=400,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(15e6),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='Gradient clipping (recommended).')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='Entropy coefficient to encourage exploration.')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon', type=int, default=10,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim-manager', type=int, default=256,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-worker', type=int, default=16,
                    help='Hidden dim for worker (k)')
parser.add_argument('--gamma-w', type=float, default=0.95,
                    help="discount factor worker")
parser.add_argument('--gamma-m', type=float, default=0.99,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.9,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=0.00001,
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')

# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='baseline',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()


def experiment(args):

    save_steps = list(torch.arange(0, int(args.max_steps),
                                   int(args.max_steps) // 10).numpy())

    logger = Logger(args.run_name, args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers, args.seed)
    feudalnet = FeudalNetwork(
        num_workers=args.num_workers,
        input_dim=envs.single_observation_space.shape,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        n_actions=envs.single_action_space.n,
        time_horizon=args.time_horizon,
        dilation=args.dilation,
        device=device,
        args=args)

    optimizer = torch.optim.RMSprop(feudalnet.parameters(), lr=args.lr,
                                    alpha=0.99, eps=1e-5)
    mask = torch.ones(args.num_workers, 1).to(device)
    goals, states = init_obj(
        args.num_workers, args.hidden_dim_manager, 2 * args.time_horizon + 1,
        device)

    x = envs.reset()
    step = 0
    while step < args.max_steps:

        # Detaching LSTMs and goals
        feudalnet.repackage_hidden()
        goals = [g.detach() for g in goals]
        storage = Storage(size=args.num_steps,
                          keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                'adv_m', 'adv_w'])

        for _ in range(args.num_steps):
            action_dist, goals, states, value_m, value_w \
                 = feudalnet(x, goals, states, mask)

            # If an environment is done, we zero out the goal
            goals[-1] = goals[-1] * mask

            # Take a step, log the info, get the next state
            action, logp, entropy = take_action(action_dist)
            x, reward, done, info = envs.step(action)
            logger.log_episode(info, step)

            mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            storage.add({
                'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i': feudalnet.intrinsic_reward(states, goals),
                'v_w': value_w,
                'v_m': value_m,
                'logp': logp.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                's_goal_cos': feudalnet.state_goal_cosine(states, goals),
                'm': mask
            })

            step += args.num_workers

        with torch.no_grad():
            *_, next_v_m, next_v_w = feudalnet(
                x, goals, states, mask, save=False)
            next_v_m = next_v_m.detach()
            next_v_w = next_v_w.detach()

        optimizer.zero_grad()
        loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feudalnet.parameters(), args.grad_clip)
        optimizer.step()
        logger.log_scalars(loss_dict, step)

        if len(save_steps) > 0 and step > save_steps[0]:
            torch.save({
                'model': feudalnet.state_dict(),
                'args': args,
                'processor_mean': feudalnet.preprocessor.rms.mean,
                'optim': optimizer.state_dict()},
                f'models/{args.env_name}_{args.run_name}_step={step}.pt')
            save_steps.pop(0)

    envs.close()
    torch.save({
        'model': feudalnet.state_dict(),
        'args': args,
        'processor_mean': feudalnet.preprocessor.rms.mean,
        'optim': optimizer.state_dict()},
        f'models/{args.env_name}_{args.run_name}_steps={step}.pt')


def main(args):
    run_name = args.run_name

    for seed in range(2):
        args.seed = seed
        args.run_name = f"{run_name}_seed={seed}"
        experiment(args)


if __name__ == '__main__':
    main(args)
