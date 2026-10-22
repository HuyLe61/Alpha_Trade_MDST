"""
Approach 3: PPO on the rebuilt environment (envs/xs_portfolio_env.py).

    python research/train_rl.py --decision-every 4 --seed 0

Same PPO code as the rest of the project, with the fixes the diagnosis called for: no entropy bonus,
a smaller starting action noise, per-decision rewards. The policy is scored with the same backtester
and measured costs as approaches 1 and 2. Results go to models/research/rl_k{K}_s{SEED}.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.actor_critic import ActorCritic
from agent.ppo import PPO
from envs.xs_portfolio_env import XSPortfolioEnv
from research import backtest as bt
from research.costs import cost_vector
from research.panel import build_panel

OUT = ROOT / 'models' / 'research'
LAG = 1


def policy_weights(policy, panel, cost, k):
    """Run the deterministic policy over a period; return per-bar target weights for the backtester."""
    env = XSPortfolioEnv(panel, cost, lag=LAG, decision_every=k, episode_length=None, random_start=False)
    obs, _ = env.reset()
    target = np.zeros_like(panel.ret)
    while True:
        with torch.no_grad():
            dist, _ = policy.forward(torch.FloatTensor(obs).unsqueeze(0))
            a = torch.tanh(dist.mean).numpy()[0]
        t0 = env.t
        obs, _, done, _, _ = env.step(a)
        target[t0:t0 + k] = env.w
        if done:
            break
    target[env.t:] = env.w
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decision-every', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iterations', type=int, default=400)
    ap.add_argument('--hidden', type=int, default=256, help='hidden units in the actor and critic')
    ap.add_argument('--early-stop', action='store_true',
                    help='score on 2025 every 20 iterations and keep the best policy (2025 then is no longer clean)')
    args = ap.parse_args()
    k, seed = args.decision_every, args.seed
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    panel = build_panel()
    cost = cost_vector(panel.tickers)
    sp = panel.splits()
    episode = {1: 1024, 4: 512, 26: 128}.get(k, 512)
    env = XSPortfolioEnv(sp['train'], cost, lag=LAG, decision_every=k, episode_length=episode)
    env.reset(seed=seed)

    ppo = PPO(env, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_epochs=10,
              batch_size=64, ent_coef=0.0, vf_coef=0.5)
    if args.hidden != 256:
        ppo.policy = ActorCritic(env.observation_space.shape[0], env.N, hidden_dim=args.hidden)
        ppo.optimizer = torch.optim.Adam(ppo.policy.parameters(), lr=3e-4, eps=1e-5)
    ppo.policy.log_std.data.fill_(-1.0)  # start with modest exploration noise (std 0.37 before tanh)
    t0 = time.time()
    if args.early_stop:
        stats = {'mean_rewards': []}
        best = (-np.inf, None, 0)
        for chunk in range(args.iterations // 20):
            stats['mean_rewards'] += ppo.train(n_iterations=20, steps_per_iter=2048)['mean_rewards']
            v = sp['val']
            sr = bt.run(policy_weights(ppo.policy, v, cost, k), v.ret, v.index.date, cost=cost, lag=LAG)['sharpe']
            print(f"  iteration {20*(chunk+1)}: val Sharpe {sr:+.2f}", flush=True)
            if sr > best[0]:
                best = (sr, {kk: vv.clone() for kk, vv in ppo.policy.state_dict().items()}, 20 * (chunk + 1))
        ppo.policy.load_state_dict(best[1])
        print(f"  kept iteration {best[2]} (val Sharpe {best[0]:+.2f})", flush=True)
    else:
        stats = ppo.train(n_iterations=args.iterations, steps_per_iter=2048)

    tag = f"rl_k{k}_s{seed}" + (f"_h{args.hidden}" if args.hidden != 256 else '') + ('_es' if args.early_stop else '')
    res = {'decision_every': k, 'seed': seed, 'hidden': args.hidden, 'early_stop': args.early_stop, 'minutes': (time.time() - t0) / 60,
           'final_mean_reward_bps': float(np.mean(stats['mean_rewards'][-20:])),
           'first_mean_reward_bps': float(np.mean(stats['mean_rewards'][:20])),
           'policy_std': float(torch.exp(ppo.policy.log_std.detach()).mean())}
    for name, p in sp.items():
        w = policy_weights(ppo.policy, p, cost, k)
        m = bt.run(w, p.ret, p.index.date, cost=cost, lag=LAG, mkt=p.ret.mean(1))
        m2 = bt.run(w, p.ret, p.index.date, cost=cost * 2, lag=LAG)
        res[name] = {key: float(v) for key, v in m.items() if key != 'net'}
        res[name]['sharpe_2x_cost'] = float(m2['sharpe'])
        print(f"{tag} {name:5s} {bt.fmt(m)}  beta {m['beta']:+.2f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=2))
    torch.save({'policy_state_dict': ppo.policy.state_dict(), 'obs_dim': env.observation_space.shape[0],
                'act_dim': env.N, 'decision_every': k, 'lag': LAG, 'market_scale': panel.market_scale},
               OUT / f'{tag}.pt')


if __name__ == '__main__':
    main()
