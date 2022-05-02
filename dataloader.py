import tensorflow as tf
from augmentation import (random_crop, noising, h_flip)

def dataloader(
        batch_size = None,
        validation_split = 0.1,
        augmentation = None,
        mode = 'train'
    ):
    for idx, aug in enumerate(augmentation):
        assert aug in ['random_crop', 'pca_noising', 'h_flip']
        if aug == 'h_flip' and idx != -1:
            augmentation[idx], augmentation[-1] = augmentation[-1], aug, 
    
    print(augmentation)

    
    if batch_size is None:
        batch_size = int(len(feature))

    def standardize(feature):
        feature = tf.cast(feature, tf.float32)
        return tf.math.divide(tf.math.subtract(feature, tf.constant([125.3069, 122.95015, 113.866])),
                              tf.constant([70.49192, 68.40918, 68.15831]))

    def training(feature, label):
        ds = tf.data.Dataset.from_tensor_slices((standardize(feature), label)).cache()
        ds = ds.shuffle(100)

        if validation_split:
            val_len = int(len(feature) * validation_split)
            val_ds = ds.take(val_len).batch(val_len, drop_remainder=True).prefetch(1)
            train_ds = ds.skip(val_len)
            ds = train_ds

        concat = [ds]
        for idx, aug in enumerate(augmentation):
            if aug == 'random_crop':
                x = ds.map(random_crop)
            elif aug == 'pca_noising':
                x = ds.map(noising)
            elif aug == 'h_flip':
                for idx, data in enumerate(concat[:3]):
                    print(idx)
                    x = data.shard(num_shards=2, index=tf.cast(tf.where(tf.random.uniform([], 0, 1)>0.5, 1, 0), 'int64')).map(h_flip)
                    concat.append(x)
            if idx < len(augmentation)-1:
                concat.append(x)
        for data in concat[1:]:
            ds = ds.concatenate(data)

        ds = ds.cache()
        ds = ds.shuffle(100).batch(batch_size = batch_size, drop_remainder = True).prefetch(1)
        if validation_split:
            return ds, val_ds
        return ds
   
    def test(feature, label):
        ds = tf.data.Dataset.from_tensor_slices((standardize(feature), label))
        ds = ds.batch(batch_size = batch_size)
        return ds

    def inference(feature):
        return standardize(feature)

    if mode == 'train':
        return training
    elif mode == 'test':
        return test
    elif mode == 'inference':
        return inference

if __name__ == '__main__':
    from tensorflow.keras.datasets.cifar10 import load_data
    (x_train, y_train), (x_test, y_test) = load_data()
    loader = dataloader(batch_size = 256,
                        validation_split = 0.3,
                        augmentation=['random_crop', 'h_flip', 'pca_noising'],
                        mode = 'train')
    train_ds, val_ds = loader(x_train, y_train)
    print(train_ds.cardinality(), val_ds.cardinality())
