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
        self.H = [0.0] * n_arms # Preferences for each arm
        self.pi = [1.0/ n_arms] * n_arms # Initial softmax probabilities

    def select_action(self) -> int:
        """Select action based on current softmax policy."""
        return np.random.choice(a=self.n_arms, p=self.pi) # returns the index of the chosen arm
        
    def update(self, action: int, reward: float) -> None:
        """Update preferences using adaptive baseline."""
        self.rewards_history.append(reward)
        baseline = self.get_baseline() # adaptive baseline
        
        for a in range(self.n_arms):
            if action == a:
                self.H[a] += self.alpha * (reward - baseline) * (1 - self.pi[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * self.pi[a]
        
        exp_H = np.exp(self.H)
        self.pi = exp_H / np.sum(exp_H)
    
    def get_baseline(self) -> float:
        """Return current baseline value."""
        if not self.rewards_history:
            return 0.0
        
        recent_rewards = [0.0]*self.window_size
        # select the recent rewards using window size
        if len(self.rewards_history) < self.window_size:
            recent_rewards = self.rewards_history
        else:
            recent_rewards = self.rewards_history[-self.window_size:]
        
        recent_rewards = np.array(recent_rewards)
        avg_reward = np.mean(recent_rewards)
        
        eps = 1e-8
        deviations = (recent_rewards - avg_reward) ** 2
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

    all_rewards = np.zeros((n_runs, n_steps))
    all_percentage_optimal_action = np.zeros((n_runs, n_steps))
    all_baselines = np.zeros((n_runs, n_steps))
    all_cumulative_regrets = np.zeros((n_runs, n_steps))
    for run in range(n_runs):
        bandit = NonStationaryBandit(alpha=alpha, beta=beta, window_size=window_size, n_arms=n_arms)
        for step in range(n_steps):
            bandit.step()
            all_rewards[run, step] = bandit.gradient_bandit.rewards_history[-1]
            all_percentage_optimal_action[run, step] = bandit.percent_optimal / (step + 1)
            all_baselines[run, step] = bandit.gradient_bandit.get_baseline()
            all_cumulative_regrets[run, step] = bandit.cumulative_regret
    
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
        self.true_means = np.random.normal(0, 1, self.n_arms) # true action values initialized from N(0,1)
        self.timestep = 0
        self.gradient_bandit = AdaptiveGradientBandit(n_arms= self.n_arms, alpha=alpha, beta=beta, window_size=window_size)
        self.cumulative_regret = 0.0
        self.percent_optimal = 0.0

    def step(self) -> None:
        """Update true means with random walk."""
        optimal_arm = np.argmax(self.true_means)
        
        action = self.gradient_bandit.select_action()
        reward = np.random.normal(self.true_means[action], 1)
        self.gradient_bandit.update(action, reward)
        self.timestep += 1

        # track matrics
        if action == optimal_arm:
            self.percent_optimal += 1
        self.cumulative_regret += self.true_means[optimal_arm] - self.true_means[action]

        # every 500 steps, update the true means using N(0,0.1)
        if self.timestep % 500 == 0 and self.timestep != 0:
            self.true_means += np.random.normal(0, 0.1, self.n_arms)
def main():
    """Run experiments and plot results."""

    np.random.seed(42)

    n_runs = 100
    n_steps = 2000
    n_arms = 10
    alpha = 0.1
    window_size = 100

    beta_values = [0.0, 0.3, 0.6]
    all_results = {}
    
    # Run experiments for all beta values
    for beta in beta_values:
        print(f"Running experiments for beta={beta}...")
        results = run_experiment(n_runs, n_steps, n_arms, alpha, beta, window_size)
        
        # Calculate running average over last 100 steps
        running_avg = np.zeros(n_steps)
        for i in range(n_steps):
            if i >= window_size - 1:
                running_avg[i] = np.mean(results['rewards'][i - window_size + 1: i + 1 ])
            else:
                running_avg[i] = np.mean(results['rewards'][:i+1])
        
        all_results[beta] = {
            'rewards': running_avg,
            'optimal_actions': results['optimal_actions'],
            'baselines': results['baselines'],
            'cumulative_regret': results['cumulative_regret']
        }
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot 1: Average Reward
    for beta in beta_values:
        axes[0].plot(all_results[beta]['rewards'], label=f'β={beta}')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Average Reward vs Steps')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Optimal Actions
    for beta in beta_values:
        axes[1].plot(all_results[beta]['optimal_actions'] * 100, label=f'β={beta}')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('% Optimal Action')
    axes[1].set_title('Percentage of Optimal Action vs Steps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Baseline Values
    for beta in beta_values:
        axes[2].plot(all_results[beta]['baselines'], label=f'β={beta}')
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Baseline Value')
    axes[2].set_title('Baseline Value vs Steps')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Cumulative Regret
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

