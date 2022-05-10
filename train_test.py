import tensorflow as tf
from tqdm import tqdm
from scheduler import CosineDecay


@tf.function
def train_loop(model, feature, label, optimizer, loss, metric):
    with tf.GradientTape() as tape:
        y_pred = model(feature, training = True)
        train_loss = loss(label, y_pred)
    trainable = model.trainable_weights
    grads = tape.gradient(train_loss, trainable)
    optimizer.apply_gradients(zip(grads, trainable))
    metric.update_state(label, y_pred)
    return train_loss

@tf.function
def test_loop(model, feature, label, loss, metric):
    y_pred = model(feature, training = False)
    val_loss = loss(label, y_pred)
    metric.update_state(label, y_pred)
    return val_loss

def train(model, train_ds, epochs, optimizer, loss, metric, regularizer = None, val_ds = None):
    log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    init_lr = Config.lr
    if regularizer:
        init_wd = Config.weight_decay
        
    for epoch in range(epochs):
        
        with  tqdm(train_ds, unit = ' step') as t_epoch:
            t_epoch.set_description(f'Epoch {epoch+1}')

            for step, (feature, label) in enumerate(t_epoch):
                train_loss = train_loop(model, feature, label, optimizer, loss, metric)
                train_acc = metric.result()                
                t_epoch.set_postfix({'loss': train_loss.numpy(), 'acc': train_acc.numpy() })

            log['loss'].append(train_loss.numpy())
            log['acc'].append(train_acc.numpy())
            metric.reset_states()
               
            if val_ds:
                for step, (feature, label) in enumerate(val_ds):
                    val_loss = test_loop(model, feature, label, loss, metric)
                    val_acc = metric.result()
                    log['val_loss'].append(val_loss.numpy())
                    log['val_acc'].append(val_acc.numpy())
                    print(f'val_loss: {val_loss}         val)acc: {val_acc}')
                metric.reset_states()

            optimizer.lr = CosineDecay(init_lr = init_lr, decay_step = 20)(epoch)
            regularizer.l2 = CosineDecay(init_lr = init_wd, decay_step = 20)(epoch)
   
    return log

def test(model, test_ds, loss, metric):
    with tqdm(test_ds, unit = ' step') as pbar:
        pbar.set_description('Test phase')
        for step, (feature, label) in enumerate(pbar):
            test_loss = test_loop(model, feature, label, loss, metric)
            test_acc = metric.result()
            metric.reset_states()
        pbar.set_postfix({'test_loss': test_loss, 'test_acc': test_acc})
