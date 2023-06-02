from infer import InferenceHelper
from models import UnetAdaptiveBins
import model_io
from PIL import Image
infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a batched rgb tensor
# example_rgb_batch = ...  
# bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)
print(bin_centers)
print(predicted_depth)
# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/media/ros/A666B94D66B91F4D/ros/learning/depth/AdaBins/out/", "path/to/output/dir/")


