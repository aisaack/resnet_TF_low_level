import tensorflow as tf

def dataloader(
        batch_size = None,
        validation_split = 0.1,
        preprocessing = None,
        mode = 'train'
    ):
    
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

        if preprocessing:
            ds = ds.apply(preprocessing).shuffle(100)

        ds = ds.batch(batch_size = batch_size, drop_remainder = True).prefetch(1)

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
