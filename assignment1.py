import numpy as np
import matplotlib.pyplot as plt

class AdaptiveGradientBandit:
    def __init__(self, n_arms: int, alpha: float, beta: float, window_size: int):
        """
        Initialize the adaptive gradient bandit agent.
        
        Args:
            n_arms: Number of arms
            alpha: Step size for preference updates
            beta: Weight for variance adaptation (0 = standard baseline)
            window_size: Number of recent rewards to consider for variance
        """
        self.n_arms = n_arms 
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.rewards_history = [] 
        self.H = [0.0] * n_arms # Numerical preferences for each arm
        # pi stores the actual selection probabilities derived from preferences via softmax
        self.pi = [1.0/ n_arms] * n_arms # Initial softmax probabilities

    def select_action(self) -> int:
        """Select action based on current softmax policy."""
        # Sample from categorical distribution defined by current probabilities
        return np.random.choice(a=self.n_arms, p=self.pi) # returns the index of the chosen arm
        
    def update(self, action: int, reward: float) -> None:
        """Update preferences using adaptive baseline."""
        # Store reward for baseline calculation
        self.rewards_history.append(reward)
        baseline = self.get_baseline() # adaptive baseline
        
        # Gradient bandit update rule
        for a in range(self.n_arms):
            if action == a:
                # For chosen arm: increase preference proportional to (reward - baseline)
                self.H[a] += self.alpha * (reward - baseline) * (1 - self.pi[a])
            else:
                # For unchosen arms: decrease preference proportional to (reward - baseline)
                self.H[a] -= self.alpha * (reward - baseline) * self.pi[a]
        
        # Update action probabilities via softmaxs
        exp_H = np.exp(self.H)
        self.pi = exp_H / np.sum(exp_H)
    
    def get_baseline(self) -> float:
        """Return current baseline value."""
        if not self.rewards_history:
            return 0.0
 
        recent_rewards = [0.0]*self.window_size
        if len(self.rewards_history) < self.window_size:
            recent_rewards = self.rewards_history
        else:
            # Use only the last 'window_size' rewards for recency
            recent_rewards = self.rewards_history[-self.window_size:]
        
        recent_rewards = np.array(recent_rewards)
        recent_rewards_mean = np.mean(recent_rewards)
        avg_reward = np.mean(self.rewards_history) #running average over all rewards
        
        eps = 1e-8
        # Calculate squared deviations from mean
        deviations = (recent_rewards - recent_rewards_mean) ** 2
        # Inverse variance weighting so rewards with lower variance get higher weight
        weights = 1.0 / (deviations + eps) # prevent division by zero
        weights  = weights/ np.sum(weights) # normalize the weights

        recent_variance_adjusted_mean = np.sum(weights * recent_rewards)
        
        return (1 - self.beta) * avg_reward + self.beta * recent_variance_adjusted_mean


def run_experiment(n_runs: int, n_steps: int, n_arms: int, alpha: float, 
                   beta: float, window_size: int) -> dict:
    """
    Run multiple experiments and return averaged results.
    
    Returns:
        dict with keys: 'rewards', 'optimal_actions', 'baselines'
    """
    # Initialize arrays to store results across all runs
    all_rewards = np.zeros((n_runs, n_steps))
    all_percentage_optimal_action = np.zeros((n_runs, n_steps))
    all_baselines = np.zeros((n_runs, n_steps))
    all_cumulative_regrets = np.zeros((n_runs, n_steps))
    
    # Run multiple runs
    for run in range(n_runs):
        # Create fresh bandit environment for each run
        bandit = NonStationaryBandit(alpha=alpha, beta=beta, window_size=window_size, n_arms=n_arms)
        for step in range(n_steps):
            # Execute one step: select action, receive reward, update
            bandit.step()
            # Record metrics for this step
            all_rewards[run, step] = bandit.gradient_bandit.rewards_history[-1]
            all_percentage_optimal_action[run, step] = bandit.optimal_choice
            all_baselines[run, step] = bandit.gradient_bandit.get_baseline()
            all_cumulative_regrets[run, step] = bandit.cumulative_regret
    
    # Average across all runs
    return {
        'rewards': np.mean(all_rewards, axis=0),
        'optimal_actions': np.mean(all_percentage_optimal_action, axis=0),
        'baselines': np.mean(all_baselines, axis=0),
        'cumulative_regret': np.mean(all_cumulative_regrets, axis=0)
    }

class NonStationaryBandit:
    """Non-stationary 10-armed bandit with random walk."""
    def __init__(self, alpha: float, beta: float, window_size: int, n_arms: int = 10 ):
        self.n_arms = n_arms
        # Initialize true action values from standard normal
        self.true_means = np.random.normal(0, 1, self.n_arms) # true action values initialized from N(0,1)
        self.timestep = 0
        self.gradient_bandit = AdaptiveGradientBandit(n_arms= self.n_arms, alpha=alpha, beta=beta, window_size=window_size)
        self.cumulative_regret = 0.0
        self.optimal_choice = 0

    def step(self) -> None:
        """Execute one step: select action, get reward, update agent, track metrics."""
        # Determine which arm is optimal
        optimal_arm = np.argmax(self.true_means)
        
        # select arm based on its current policy
        action = self.gradient_bandit.select_action()
        # Reward is sampled from normal distribution centered at true mean, std of 1 adds some noise
        reward = np.random.normal(self.true_means[action], 1)
        # Update agent's preferences based on observed reward
        self.gradient_bandit.update(action, reward)
        self.timestep += 1

        self.optimal_choice = int(action == optimal_arm)
        self.cumulative_regret += self.true_means[optimal_arm] - self.true_means[action]

        # Every 500 steps, the true means shift via random walk
        if self.timestep % 500 == 0 and self.timestep != 0:
            # Add Gaussian noise with variance 0.1 to each true mean
            self.true_means += np.random.normal(0, np.sqrt(0.1), self.n_arms)

def main():
    """Run experiments and plot results."""
    # Set seed for reproducibility
    np.random.seed(42)

    n_runs = 100      # Number of independent runs to average over
    n_steps = 2000    # Time horizon for each run
    n_arms = 10       # Number of bandit arms
    alpha = 0.1       # Learning rate for preference updates
    window_size = 100 # Size of sliding window for baseline calculation
    beta_values = [0.0, 0.3, 0.6] # Different beta values to test
    all_results = {}
    
    # Run experiments for all beta values
    for beta in beta_values:
        print(f"Running experiments for beta={beta}...")
        results = run_experiment(n_runs, n_steps, n_arms, alpha, beta, window_size)
        
        running_avg = np.zeros(n_steps)
        for i in range(n_steps):
            if i >= window_size - 1:
                # Use full window once we have enough data
                running_avg[i] = np.mean(results['rewards'][i - window_size + 1: i + 1 ])
            else:
                # Use all available data when we don't have full window yet
                running_avg[i] = np.mean(results['rewards'][:i+1])
        
        all_results[beta] = {
            'rewards': running_avg,
            'optimal_actions': results['optimal_actions'],
            'baselines': results['baselines'],
            'cumulative_regret': results['cumulative_regret']
        }
    
    # plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Average Reward
    for beta in beta_values:
        axes[0].plot(all_results[beta]['rewards'], label=f'β={beta}')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Average Reward vs Steps')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Optimal Actions
    for beta in beta_values:
        axes[1].plot(all_results[beta]['optimal_actions'] * 100, label=f'β={beta}')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('% Optimal Action')
    axes[1].set_title('Percentage of Optimal Action vs Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Baseline Values
    for beta in beta_values:
        axes[2].plot(all_results[beta]['baselines'], label=f'β={beta}')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Baseline Value')
    axes[2].set_title('Baseline Value vs Steps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Cumulative Regret
    for beta in beta_values:
        axes[3].plot(all_results[beta]['cumulative_regret'], label=f'β={beta}')
    axes[3].set_xlabel('Steps')
    axes[3].set_ylabel('Cumulative Regret')
    axes[3].set_title('Cumulative Regret vs Steps')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("adaptive_gradient_bandit_results.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()