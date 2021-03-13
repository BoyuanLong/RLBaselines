from trainer import Trainer

def main(args):
    trainer = Trainer(args)
    trainer.train()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--n_iter', type=int, default=2000)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    args = parser.parse_args()
    main(args)