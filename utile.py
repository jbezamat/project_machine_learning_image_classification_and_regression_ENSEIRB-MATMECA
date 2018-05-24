from mp1 import *

def test_show_generate():
    im1 = generate_a_rectangle(10, True)
    plt.imshow(im1.reshape(100,100), cmap='gray')

    plt.show()

    im2 = generate_a_disk(10, True)
    plt.imshow(im2.reshape(100,100), cmap='gray')

    plt.show()

    [im3, v] = generate_a_triangle(20, True)
    plt.imshow(im3.reshape(100,100), cmap='gray')

    plt.show()
