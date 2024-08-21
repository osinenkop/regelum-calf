python goalagent/sac.py \
    seed=1,2,3,4,5,6,7,8,9,10 \
    env=lunar_lander \
    --experiment=sac_lunar_lander \
    autotune=False \
    alpha=0.001 \
    gamma=0.993 \
    policy_lr=0.001 \
    q_lr=0.001