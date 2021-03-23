from genericpath import exists
from Trainers.offline_trainer import OfflineTrainer
from Trainers.trainer import Trainer
from utils.utils import global_seed
import os, time

def main(args):
    global_seed(args.seed)
    if args.mode == 'online':
        trainer = Trainer(args)
    else:
        trainer = OfflineTrainer(args)
        trainer.generate_buffer()
    trainer.train()
    trainer.test()
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--mode', type=str, default='online', choices=['online', 'offline'])
    parser.add_argument('--save_model', type=bool ,default=False)

    parser.add_argument('--train_iter', type=int, default=20000)
    parser.add_argument('--test_iter', type=int, default=1000)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=50)

    parser.add_argument('--collect_policy', type=str, default='agent')
    parser.add_argument('--render_mode', type=str, default='human')

    # Q-Learning hparams
    parser.add_argument('--e_greedy', type=bool, default=True)
    parser.add_argument('--e_decay_rate', type=float, default=1e-6)

    # Buffer
    parser.add_argument('--buffer_size', type=int, default=10000)

    # Logging
    parser.add_argument('--log_path', type=str, default='./logging')

    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.log_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = "{}_{}_{}".format(args.env, args.mode, time.strftime("%d-%m-%Y_%H-%M-%S"))
    logdir = os.path.join(data_path, logdir)
    assert not (os.path.exists(logdir))
    os.makedirs(logdir)
    args.logdir = logdir

    main(args)
