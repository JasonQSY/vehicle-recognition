# rob535-perception-proj

ROB 535 Perception Project

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
python train.py -e sn_full -t sn
```

To continue training model `sn_full`,

```bash
python train.py -c sn_full -e sn_full -t sn
```

The training code would automatically save `${model}_${epoch}` under `exp`. For example, if we train a model `sn_full` for 10 epochs, there would be `sn_full_1`, `sn_full_2`, etc. under `exp`. These snapshots are used for validation.

## Evaluation

To generate the prediction, run

```bash
rm -rf save
mkdir save
python generate.py -c sn_full -e sn_full -t sn
```

## Ensemble

Besides training and evaluation, we want to submit an ensemble of ConvNets to improve performance. These can be done by

```bash
python ensemble.py
```


## Kaggle Submission

See https://github.com/Kaggle/kaggle-api

In general,

```
kaggle competitions submit -c fall2018-rob535-task1 -f result.csv -m "msg"
```

