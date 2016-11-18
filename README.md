### MinTF

A thin wrapper around TensorFlow for concise model definition. Also takes care of exporting trained models suitable on embedded devices. Highly influenced by [Keras](https://keras.io/) and [TFLearn](http://tflearn.org/)

### An MNIST Sample

Full source available in [samples](examples/mnist)

#### Defining Models
```python
mnist_model = Model()
mnist_model.add(Dense([784, 10], activation=tf.nn.softmax, name='output'))
```

#### Training Models
```python
trainer = Trainer(mnist_model, [784], [10], loss, accuracy, MNISTData())
trainer.train(tf.train.GradientDescentOptimizer(0.01))
```

#### Exporting Trained Models
```python
trainer.export('mnist.pb')
```

#### Predicting Using Trained Models
```python
predictions = Predictor('mnist.pb').predict(mnist.test.images)
```
