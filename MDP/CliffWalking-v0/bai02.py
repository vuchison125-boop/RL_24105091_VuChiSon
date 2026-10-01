import gymnasium as gym

env = gym.make("CliffWalking-v1")

print("Bellman Optimality Equation:")
print("V(s) = max_a Σ P(s'|s,a)[R + γV(s')]")
print()
print("max_a Q(s,a):")
print("Chọn action có giá trị kỳ vọng lớn nhất.")
print()
print("Gamma (γ):")
print("γ quyết định mức độ quan tâm đến reward trong tương lai.")

env.close()