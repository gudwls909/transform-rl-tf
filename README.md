# transform-rl-tf

## Usage
you can currently train the RL model in
- one of the `[ppo,ddpg]` model by `--algorithm` argument
- one of the `['r','rsc','rsh','rss','rsst']` mode by `--env` argument
  - `r` refers to 'rotate'
  - `rsc` refers to 'rotate, scale'
  - `rsh` refers to 'rotate, shear'
  - `rss` refers to 'rotate, shear, scale'
  - `rsst` refers to 'rotate, shear, scale, translate'
<br>

<b>to train the model (MNIST)</b>
```
$ python main.py --algorithm=ppo --gpu_number=0 --epochs=1 --save_dir=r_save --env=r
```

for real-world dataset,

```
$ python main.py --algorithm=ppo --gpu_number=0 --epochs=1 --save_dir=r_save --env=r --data_type=cifar10
```

after running the command, 
- dataset in reference to the affined (rotate in the upper case) MNIST is generated in `data` directory
- all the savings(such as image or checkpoint) are saved in `save/ppo/r_save` directory
<br>

<b>to continue train with the checkpoint in the `r_save` directory</b>
```
$ python main.py --algorithm=ppo --gpu_number=0 --epochs=1 --save_dir=r_save --env=r --continue_train
```
<br>

<b>to test the model</b>
```
$ python main.py --algorithm=ppo --gpu_number=0 --save_dir=r_save --env=r --test
```

