# Постобработка OCR паспорта РФ

```
python postprocess.py data/test_predictions.jsonl answers.jsonl   # инференс (CPU, без интернета)
python train.py                                                    # пересобрать model/ и увидеть out-of-fold кривую
python evaluate.py answers.jsonl data/train_labels.jsonl          # сверка (для train; на train оценка завышена)
```

Нужны `data/` и `reference/subdivisions.csv` из задания. Код: `core.py` (MRZ, слияние, срок действия),
`features.py` (признаки и словари), `postprocess.py` (решение), `train.py` (обучение).
`model/model.pkl` собран на scikit-learn 1.8; при другой версии выполните `python train.py` (около минуты).
Отчёт: `report.md`.
