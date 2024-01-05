import matplotlib.pyplot as plt
from pytorch_fid import fid_score


def save_array_as_image(array, filename):
    plt.imshow(array, cmap='hot')  # You can change the colormap as needed
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def calculate_fid(real_images, generated_images):
    fid = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=50, device='cuda', dims=2048)
    return fid