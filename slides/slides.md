---
header-includes: |
    \usepackage[utf8]{inputenc}
---

# Fashion MNIST

![Fashion MNIST image example](fashion-mnist-sprite.png)


# Fashion MNIST Definition

- The original MNIST dataset has been used as testing dataset for a lot of learning
    methods for a long time, however:
    + It is too easy. With CNNs you can achieve a 99.7% of accuracy.
    + It has been overused.
    + It is not representative of the modern tasks.
- Therefore, the guys from Zalando have created a new dataset, 
    [FashionMNIST](https://arxiv.org/abs/1708.07747) with similar 
    characteristics to MNIST:
    + 60000+10000 28x28 greyscale images and  10 classes 

| Label | Description | Label | Description |
|-------|-------------|-------|-------------|
| 0     | T-shirt/top | 5     | Sandal      |
| 1     | Trouser     | 6     | Shirt       |
| 2     | Pullover    | 7     | Sneaker     |
| 3     | Dress       | 8     | Bag         |
| 4     | Coat        | 9     | Ankle boot  |


# Kuzujishi MNIST 

![Kuzujishi MNIST image example](kuzumnist_examples.png)

# Kuzujishi MNIST Definition

+ [Available in Github](https://github.com/rois-codh/kmnist)
+ Dataset used in **Deep Learning for Classical Japanese Literature** _Tarin
  Clanuwat et al._ 2018
+ 60000+10000 28x28 greyscale images and 10 classes 
+ Loading the data requires specific functions that are provided for the course
  (LeCun's original MNIST data format)


| Label | Description | Label | Description |
|-------|-------------|-------|-------------|
| 0     | お - o      | 5     | は - ha     |
| 1     | き - ki     | 6     | ま - ma     |
| 2     | す - su     | 7     | や - ya     |
| 3     | つ - tsu    | 8     | れ - re     |
| 4     | な - na     | 9     | を - wo     |

