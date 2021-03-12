from trainer import Trainer

def main():
    trainer = Trainer('Pendulum-v0')
    t = 0
    while t < 100:
        trainer.sample_trajectory(True, ('rgb_array'))
        t += 1
    

if __name__ == "__main__":
    main()