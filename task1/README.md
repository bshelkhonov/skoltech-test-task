# Модификация кода для генерации карт

Я взял датасет [EuroSAT](https://www.kaggle.com/datasets/1f15fbfaff64d4fba50c4c333c8b07831e0deff5d980d214343d4fc4842758b4) (все фотографии, кроме озёр).

Написал свой класс датасета для него в файле [dataset.py](./dataset.py). (Если бы у меня был python поновее, можно было бы взять встроенный EuroSat в torch).
Файл [train_maps.py](./train_maps.py) аналогичен остальным. Только я добавил рандомные повороты в качестве аугментаций.

В папку [contents](./contents) поместил пару примеров генерируемых изображений. Их можно отличить по наличию слова _maps_ в названии. Если что, два верхних ряда изображений -- сгерерированных. Два нижних -- из оригинального датасета для сравнения. 


# minDiffusion

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_19.png">
</p>

Goal of this educational repository is to provide a self-contained, minimalistic implementation of diffusion models using Pytorch.

Many implementations of diffusion models can be a bit overwhelming. Here, `superminddpm` : under 200 lines of code, fully self contained implementation of DDPM with Pytorch is a good starting point for anyone who wants to get started with Denoising Diffusion Models, without having to spend time on the details.

Simply:

```
$ python superminddpm.py
```

Above script is self-contained. (Of course, you need to have pytorch and torchvision installed. Latest version should suffice. We do not use any cutting edge features.)

If you want to use the bit more refactored code, that runs CIFAR10 dataset:

```
$ python train_cifar10.py
```

<!-- #region -->
<p align="center">
<img  src="contents/_ddpm_sample_cifar43.png">
</p>

Above result took about 2 hours of training on single 3090 GPU. Top 8 images are generated, bottom 8 are ground truth.

Here is another example, trained on 100 epochs (about 1.5 hours)

<p align="center">
<img  src="contents/_ddpm_sample_cifar100.png">
</p>

Currently has:

- [x] Tiny implementation of DDPM
- [x] MNIST, CIFAR dataset.
- [x] Simple unet structure. + Simple Time embeddings.
- [x] CelebA dataset.

TODOS

- [ ] DDIM
- [ ] Classifier Guidance
- [ ] Multimodality

# Updates!

- Using more parameter yields better result for MNIST.
- More comments in superminddpm.py
