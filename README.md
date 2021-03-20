# RLBaselines

### Q Learning w/ epsilon-greedy
```
python main.py --env "FrozenLake-v0" --learning_rate 0.1 --collect_policy agent --render_mode none --e_decay_rate 1e-5
```

### Q Learning w/ random exploration
```
python main.py --env "FrozenLake-v0" --learning_rate 0.1 --collect_policy random --render_mode none --e_greedy False
```