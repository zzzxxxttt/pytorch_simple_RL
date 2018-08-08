# pytorch_rl_iof

Reinforcement learning algorithms in one file

DDPG for MountainCarContinuous-v0:
![image](https://github.com/zzzxxxttt/pytorch_rl_iof/blob/master/figures/ddpg_mtcar.png)

Code mainly adopted from https://github.com/lirnli/OpenAI-gym-solutions/blob/master/Continuous_Deep_Deterministic_Policy_Gradient_Net/DDPG%20Class%20ver2.ipynb
I just rewrite the code using pytorch, and fixed a small bug in the original code when updating the actor.

A2C for MountainCarContinuous-v0:
![image](https://github.com/zzzxxxttt/pytorch_rl_iof/blob/master/figures/a2c_mtcar.png)

Code mainly adopted from https://gist.github.com/Ashioto/10ec680395db48ddac1ad848f5f7382c#file-actorcritic-py

I just rewrite the code using pytorch.

As mentioned by the original author, this a2c code is extremely unstable, if you find it does not work, just kill the process and restart training :)