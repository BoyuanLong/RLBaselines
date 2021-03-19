from trainer import Trainer
from offline_trainer import Trainer

def main(args):
    trainer = Trainer(args)
    # trainer.train()
    trainer.test()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--train_iter', type=int, default=20000)
    parser.add_argument('--test_iter', type=int, default=1000)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--collect_policy', type=str, default='agent')
    parser.add_argument('--render_mode', type=str, default='human')

    # Q-Learning hparams
    parser.add_argument('--e_greedy', type=bool, default=True)
    parser.add_argument('--e_decay_rate', type=float, default=1e-6)

    args = parser.parse_args()
    main(args)