# RLBaselines

### Q Learning w/ epsilon-greedy
```
python main.py --env "FrozenLake-v0" --learning_rate 0.1 --collect_policy agent --render_mode none --e_decay_rate 1e-5
```

### Q Learning w/ random exploration
```
python main.py --env "FrozenLake-v0" --learning_rate 0.1 --collect_policy random --render_mode none --e_greedy False
```

### Offline Q Learning (SARSA update)
```
python main.py --env "FrozenLake8x8-v0" --learning_rate 0.1 --collect_policy agent --render_mode none --e_decay_rate 1e-5 --mode offline
```

### Vanilla REINFORCE
```
python main.py --learning_rate 0.001 --batch_size 1000 --collect_policy agent --render_mode none --train_iter 100 --mode online --seed 0 --gamma 1 --agent reinforce
```

### Vanilla AC
```
python main.py --learning_rate 0.001 --batch_size 1000 --collect_policy agent --render_mode none --train_iter 100 --mode online --seed 0 --gamma 1 --agent ac --concat_rewards
```
