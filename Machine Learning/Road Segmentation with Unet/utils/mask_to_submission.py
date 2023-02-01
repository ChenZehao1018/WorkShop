import numpy as np
import re

# assign a label to a patch
def patch_to_label(patch, foreground_threshold=0.25): 
    
    df = np.mean(patch)
    if df > foreground_threshold:
        # Road
        return 1
    else:
        # Background
        return 0

#Read a single image and outputs the strings that should go into the submission file
def mask_to_submission_strings(image_filename, pred, foreground_threshold):
    
    img_number = int(re.search(r"\d+", image_filename).group(0))
    patch_size = 16
    for j in range(0, pred.shape[1], patch_size):
        for i in range(0, pred.shape[0], patch_size):
            patch = pred[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, foreground_threshold)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)

#Convert the mask into a submission file
def masks_to_submission(submission_filename, foreground_threshold, image_filenames):
    
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for img_filename, pred in image_filenames:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(img_filename, pred, foreground_threshold))
