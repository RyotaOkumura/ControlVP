from datasets import load_from_disk


def find_most_similar_image(dataset_path: str, idx: int):
    dataset = load_from_disk(dataset_path)
    data = dataset[idx]
    image = data["image"]
    conditioning = data["conditioning"]
    image.save("image.png")
    conditioning.save("conditioning.png")


if __name__ == "__main__":
    dataset_path = "/srv/datasets3/HoliCity/dataset_w_vpts"
    idx = 16732
    find_most_similar_image(dataset_path, idx)
