import tensorflow as tf
from sklearn.decomposition import PCA

def random_crop(img, *args):
    size = tf.shape(img)
    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
    img = tf.image.random_crop(img, size)
    return [img] + [arg for arg in args]

def noising(img, *args):
    size = tf.sahpe(img)
    pca = PCA(size[0] svd_solver='full')
    noise = np.random.normal(0, 0.1, size[:2])
    noise = pca.fit(noise).components_
    noise = tf.stack([noise]*3, axis=-1)
    noise = tf.cast(noise, tf.float32)
    return [img + noise] + [arg for arg in args]

def h_flip(img, *args):
    img = tf.image.random_flip_left_right(img)
    return [img] + [arg for arg in args]
