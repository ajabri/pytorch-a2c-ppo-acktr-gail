# taskset -c 81-120 python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 100 --ops --note 'first version' --obs-interval 2
#
#
#
# python main.py --env-name "CartPole-v1" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits

# python main.py --ops --seed 222 --bonus1 0
# python main.py --ops --seed 222 --bonus1 0.01
# python main.py --ops --seed 222 --bonus1 0.02
# python main.py --ops --seed 222 --bonus1 0.005


taskset -c 201-240 python main.py --env-name 'MiniGrid-Dynamic-Obstacles-8x8-v0' --ops --seed 222
taskset -c 201-240 python main.py --env-name 'MiniGrid-Dynamic-Obstacles-16x16-v0' --ops --seed 222

# taskset -c 201-240 python main.py --env-name 'MiniGrid-Dynamic-Obstacles-6x6-v0' --ops --seed 222 --bonus1 0.04 --no-bonus 80
# taskset -c 201-240 python main.py --env-name 'MiniGrid-Dynamic-Obstacles-6x6-v0' --ops --seed 222 --bonus1 0.01 --no-bonus 80
# taskset -c 201-240 python main.py --env-name 'MiniGrid-Dynamic-Obstacles-6x6-v0' --ops --seed 222 --bonus1 0.03 --no-bonus 80
