# Diversified Semantic Distribution Matching for Dataset Distillation (MM 2024)

## Run

1. prepare pre-trained models

```
python pre_train_model.py --reproduce  -d [dataset]
```
2. distilling process 

```
python DSDM.py  --reproduce -d [dataset] -f 2 --ipc [instance/class]
```

## Condensation evaluation

Full synthetic-data evaluation retrains an evaluator and is much more expensive
than a condensation update. The project pipeline therefore evaluates at sparse
milestones by default:

```text
100, 500, 1000, 2000, 3000, 5000, 7500, 10000
```

Set `distillation.evaluate_iterations` in YAML to customize the checkpoints. An
empty list restores the legacy fixed interval from `distillation.evaluate_iter`.
The final condensation iteration is always evaluated.

## Acknowledgements

Our code in this project is built upon the work of [IDC](https://github.com/snu-mllab/efficient-dataset-condensation). We thank them for their excellent work.
