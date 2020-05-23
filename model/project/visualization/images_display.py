import numpy as np

def join_images(samples):
    images, outputs, output_thresholds, labels = samples
    sample_images = []
    for i in range(len(images)):
        image, output, output_threshold, label = images[i], outputs[i], output_thresholds[i], labels[i]
        output = np.stack((output,)*3, axis=0)
        output_threshold = np.stack((output_threshold,)*3, axis=0)
        label = np.stack((label,)*3, axis=0)
        sample_images.append(np.concatenate((image, output, output_threshold, label), 1))

    sample_output = np.concatenate(tuple(sample_images), -1)
    return sample_output

def join_image_batches(samples):
    images = [join_images(x).transpose(1,2,0) for x in samples]
        
    hight = max(img.shape[0] for img in images)
    width = sum(img.shape[1] for img in images)
    joined_images = np.zeros((hight, width, 3), dtype=np.float)
    pos = 0
    for img in images:
        joined_images[:img.shape[0], pos:pos+img.shape[1]] = img
        pos += img.shape[1]
    return joined_images