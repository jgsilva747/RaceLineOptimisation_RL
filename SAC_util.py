import d3rlpy
import gym

# Create environment
env = gym.make(("CartPole-v1"))

# Create SAC agent
sac = d3rlpy.algos.SACConfig().create(device=False)
sac.create_impl(env.observation_space, env.action_space)

print(sac)

# Training loop
for epoch in range(100):
    # Initialize episode
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        print("Step")
        # Select action based on current state
        action = sac.predict([state])[0]

        # Execute action in the environment
        next_state, reward, done, _ = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Store the experience in the replay buffer
        sac.add_experience(state, action, reward, next_state, done)

        # Move to the next state
        state = next_state

        # Perform SAC updates if the replay buffer has enough experiences
        if len(sac.replay_buffer) >= sac.batch_size:
            sac.update()

    # Print the total reward achieved in this epoch
    print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")

# Save the trained policy
sac.save_policy("trained_sac_policy")