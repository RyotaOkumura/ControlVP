from datasets import load_from_disk


def show_dataset(dataset_path: str, idx: int):
    dataset = load_from_disk(dataset_path)
    print(len(dataset))
    # data = dataset[idx]
    # image = data["image"]
    # conditioning = data["conditioning"]
    # vpts = data["vanishing_points"]
    # print(vpts)
    # image.save(f"image_{idx}.png")
    # conditioning.save(f"conditioning_{idx}.png")


if __name__ == "__main__":
    dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts"
    for idx in range(1):
        show_dataset(dataset_path, idx)
