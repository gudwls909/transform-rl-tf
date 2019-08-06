import os
import argparse
import tensorflow as tf

from agent import Agent

# print(sys.executable)
# parameter 저장하는 parser
parser = argparse.ArgumentParser(description="Pendulum")
parser.add_argument('--gpu_number', default='0', type=str)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--discount_factor', default=0.95, type=float)
parser.add_argument('--continue_train', default=False, type=bool)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--save_dir', default='ex', type=str)
parser.add_argument('--render_dir', default='render_train', type=str)
parser.add_argument('--play_dir', default='render_test', type=str)
args = parser.parse_args()

# make directories
args.save_dir = os.path.join('save', args.save_dir)
args.render_dir = os.path.join(args.save_dir, args.render_dir)
args.play_dir = os.path.join(args.save_dir, args.play_dir)
if not os.path.exists(args.render_dir):
    os.makedirs(args.render_dir)
if not os.path.exists(args.play_dir):
    os.makedirs(args.play_dir)

config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
config.log_device_placement = False
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    # 학습 or 테스트
    with tf.Session(config=config) as sess:
        agent = Agent(args, sess)

        agent.train()
        agent.save()
        
        agent.load()
        agent.play()
