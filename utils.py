import matplotlib.pyplot as plt

def display_image(image, title):
    plt.figure()
    plt.axis("off")
    plt.imshow(image.permute(1, 2, 0))
    plt.title(title)