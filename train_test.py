import tensorflow as tf

def train(model, train_ds, epochs, optimizer, loss, metric, val_ds=None):

    if not isinstance(metric, list):
        metric = list(metric)

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        for step, (feature, label) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                y_pred = model(feature, training = True)
                train_loss = loss(label, y_pred)
            trainable = model.trainable_weights
            grads = tape.gradient(train_loss, trainable)
            optimizer.apply_gradients(zip(grads, trainable))

            for m in metric:
                m.update_state(label, y_pred)
                    
            train_acc = {f'{m.name}':m.result().numpy() for m in metric}
            train_acc_name = f'At step {step+1} loss: {train_loss:.4f} '
            for key, value in train_acc.items():
                train_acc_name += f'{key}: {value:.4f} '
            print(f'{train_acc_name}')

        for m in metric:
            m.reset_states()

        if val_ds:
            for val_step, (val_feature, val_label) in enumerate(val_ds):
                val_pred = model(val_feature, training = False)
                val_loss = loss(val_label, val_pred)
                for m in metric:
                    m.update_state(val_label, val_pred)
            val_acc = {f'{m.name}': m.result().numpy() for m in metric}
            val_acc_name = f'At epoch {epoch} val_loss: {val_loss:.4f} val_'
            for key, value in val_acc.items():
                val_acc_name += f'{key}: {value:.4f} '

            print(f'{val_acc_name}\n')
            for m in metric:
                m.reset_states()


def test(model, test_ds, optimizer, loss, metric):
    
    if not isinstance(metric, list):
        metric = list(metric)

    for step, (feature, label) in enumerate(test_ds):
        test_pred = model(feature, training = False)
        test_loss = loss(label, test_pred)
        for m in metric:
            m.update_state(label, test_pred)
    test_acc = {m.name: m.result().numpy() for m in metric}
    name_base = f'test_acc: {test_acc} '
    for key, value in test_acc.items():
        name_base += f'{key}: {value:.4f}'
    
    print(f'{name_base}')
    for m in metric:
        m.reset_states()                
