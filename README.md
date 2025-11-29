# ControlVP : Interactive Geometric Refinement of AI-Generated Images with Consistent Vanishing Points

A user-guided framework for correcting vanishing point (VP) inconsistencies in AI-generated images using building contours as conditions.

![teaser](assets/readme/teaser.png)


## Installation

### Prerequisites
- Python 3.11+
- CUDA 12.1
- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```
## Training the model
### Download the training dataset
Install the HoliCity dataset from the [homepage](https://holicity.io/).
The parent path is arbitrary, but the dataset should be organized as follows:
```
|--images
|--normal_map
|--planes
|--vanishing_points
```

### Add building outlines to the dataset 
```bash
./src/script/create_training_dataset.sh
```

### Run the training script
```bash
./src/script/train_controlnet_vp_loss.sh
```
