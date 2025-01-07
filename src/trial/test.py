import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid_threshold():
    # データ準備
    x = np.linspace(0.0, 1.0, 1000)
    threshold = 1.0  # 0を中心に表示
    temperatures = [1, 5, 10, 20]
    plt.figure(figsize=(10, 6))
    # 各temperatureでのシグモイド関数をプロット
    for temp in temperatures:
        y = 2 / (1 + np.exp(-temp * (x - threshold)))
        plt.plot(x, y, label=f"temperature = {temp}")
    # 二値化関数（step関数）も表示
    plt.plot(x, x > threshold, "--", label="step function", alpha=0.5)
    plt.grid(True)
    plt.xlabel("cos θ - threshold")
    plt.ylabel("weight")
    plt.title("Sigmoid Function with Different Temperatures")
    plt.legend()
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.savefig("sigmoid_comparison.png")
    plt.close()


plot_sigmoid_threshold()
