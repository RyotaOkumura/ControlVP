import kornia
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

path_img = (
    "/srv/datasets3/HoliCity/images/2008-07/8heFyix0weuW7Kzd6A_BLg_LD_030_41_imag.jpg"
)
img = Image.open(path_img)
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
img_tensor = image_transforms(img)
img_tensor = img_tensor.unsqueeze(0)
canny = kornia.filters.Canny()
x_magnitude, x_canny = canny(img_tensor)

img_magnitude = kornia.tensor_to_image(x_magnitude.byte())
img_canny = kornia.tensor_to_image(x_canny.byte())

canny_w_thresh = kornia.filters.Canny(low_threshold=0.5, high_threshold=0.8)
x_magnitude_w_thresh, x_canny_w_thresh = canny_w_thresh(img_tensor)
img_magnitude_w_thresh = kornia.tensor_to_image(x_magnitude_w_thresh.byte())
img_canny_w_thresh = kornia.tensor_to_image(x_canny_w_thresh.byte())

# Create the plot
fig, axs = plt.subplots(1, 3, figsize=(16, 16))
axs = axs.ravel()

axs[0].axis("off")
axs[0].set_title("image source")
axs[0].imshow(np.array(img))

axs[1].axis("off")
axs[1].set_title("canny edges")
axs[1].imshow(img_canny, cmap="Greys")

axs[2].axis("off")
axs[2].set_title("canny edges with threshold")
axs[2].imshow(img_canny_w_thresh, cmap="Greys")

plt.savefig("kornia_canny.png")
