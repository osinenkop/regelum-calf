python goalagent/sac.py \
    seed=1,2,3,4,5,6,7,8,9,10 \
    env=pendulum_quanser \
    --experiment=sac_pendulum \
    policy_lr=0.00079 \
    q_lr=0.00025 \
    autotune=False alpha=0.0085