# Vehicle Recognition from a Single Image

Code for reproducing the results in our ROB 535 Self-Driving Cars Perception Project, 2018 Fall.

## Set up

- pytorch
- opencv
- numpy
- scipy
- imageio
- tqdm

## Training

To set up,

```bash
mkdir exp
```

To start training a new model,

```bash
python train.py [-c checkpoint] -e experiment -t task

# train a model for task 1 from scratch
python train.py -e resnet50_bs16_2e-5_aug -t t1

# continue training from checkpoint resnet50_bs16_2e-5_aug_5 for task 1
python train.py -c resnet50_bs16_2e-5_aug_5 -e resnet50_bs16_2e-5_aug -t t1

# train a model for task 2 from scratch
python train.py -e resnet50_t2 -t t2
```

The training code would automatically save `${model}_${epoch}` under `exp`. For example, if we train a model `resnet` for 10 epochs, there would be `resnet_1`, `resnet_2`, etc. under `exp`. These snapshots are used for validation.

## Evaluation

To generate the prediction, check `generate.py`. It should store results into a `result.csv` or `task2.csv` file. You need specify checkpoint and task.

```bash
python generate.py -c resnet50_bs16_2e-5_aug_5 -t t1
```


## Kaggle Submission

See https://github.com/Kaggle/kaggle-api

In general,

```
kaggle competitions submit -c fall2018-rob535-task1 -f result.csv -m "msg"
```

