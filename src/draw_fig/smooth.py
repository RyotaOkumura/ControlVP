def my_tb_smooth(
    scalars: list[float], weight: float
) -> list[float]:  # Weight between 0 and 1
    """

    ref: https://stackoverflow.com/questions/42011419/is-it-possible-to-call-tensorboard-smooth-function-manually

    :param scalars:
    :param weight:
    :return:
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed: list = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def plot_loss(csv_path: str, weight: float = 0.992, max_step: int = 4000):
    """損失の推移をプロットして保存する関数

    Args:
        csv_path: 損失値が保存されているCSVファイルのパス
        weight: 平滑化の重み (0-1の間)
        max_step: プロットする最大ステップ数
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    # max_stepまでのデータに制限
    df = df[df["Step"] <= max_step]

    # プロットの設定
    plt.figure(figsize=(8, 7))

    # # オリジナルの損失値を薄いグレーでプロット
    # plt.plot(df["Step"], df["Value"], color="lightgray", alpha=0.5, label="Original")

    # 平滑化した損失値を黒でプロット
    smoothed = my_tb_smooth(df["Value"].tolist(), weight)
    plt.plot(df["Step"], smoothed, color="black")

    # グラフの設定
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.tick_params(axis="both", labelsize=20)  # 目盛りの数字を14ptに拡大

    # plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    # plt.legend()

    # y軸の範囲を設定
    plt.ylim(0.354, 0.378)

    # グラフを保存
    plt.savefig("training_loss.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # 使用例
    plot_loss("training_loss.csv")
