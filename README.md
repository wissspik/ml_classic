# Santander Customer Transaction Prediction

Решение для соревнования Kaggle: [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction).

## Задача

Нужно предсказать вероятность того, что клиент совершит транзакцию. Это задача бинарной классификации:

- `target = 1` - клиент совершит транзакцию;
- `target = 0` - клиент не совершит транзакцию.

Основная метрика соревнования - `ROC-AUC`.

## Данные

Используются файлы:

- `train.csv` - обучающая выборка;
- `test.csv` - тестовая выборка без таргета;
- `sample_submission.csv` - шаблон отправки.

В данных 200 числовых анонимизированных признаков `var_0` ... `var_199`. Поле `ID_code` не использовалось как признак. Целевая переменная несбалансирована: положительный класс составляет около 10%.

## Что было сделано

1. Проведена базовая EDA:
   - проверены размеры и типы данных;
   - изучены описательные статистики признаков;
   - построено распределение целевой переменной.

2. Подготовлены признаки:
   - выделены все признаки вида `var_*`;
   - исключён `ID_code`;
   - для линейных моделей применялся `StandardScaler`;
   - отдельно проверялась L1-регуляризация для отбора слабых признаков.

3. Обучены несколько моделей:
   - `LogisticRegression`;
   - `DecisionTreeClassifier`;
   - `CatBoostClassifier`;
   - `LightGBM`;
   - `XGBoost`.

4. Пробовался подбор гиперпараметров:
   - `GridSearchCV` для дерева и бустингов;
   - `RandomizedSearchCV` для логистической регрессии;
   - регуляризация `L1`, `L2`, `ElasticNet` для линейной модели.

5. Для финальных предсказаний использовались вероятности класса `1`:

```python
predictions = model.predict_proba(X_test)[:, 1]
```

## Лучший результат

Лучший сохранённый submission:

```text
result/catboost_strong_baseline.csv
```

На скриншоте Kaggle для `catboost_strong_baseline.csv` зафиксирован результат:

```text
0.89507 / 0.89773
```

## Что сработало лучше всего

Лучше всего сработал CatBoost с большим числом итераций, небольшой скоростью обучения и early stopping. Бустинги оказались сильнее одиночного дерева и линейной модели.

Использованный сильный baseline:

```python
CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="PRAUC",
    iterations=5000,
    learning_rate=0.03,
    depth=12,
    l2_leaf_reg=10,
    random_seed=42,
    task_type="GPU"
)
```

## Что не сработало или было хуже

- Обычное дерево решений без настройки дало заметно более слабый результат.
- Полный `GridSearchCV` для тяжёлых моделей оказался слишком долгим.
- Слишком глубокие и долгие бустинги требуют аккуратного контроля переобучения.
- Линейные модели требуют масштабирования признаков, иначе возможны проблемы со сходимостью.
- L1-регуляризация может занулять слишком много признаков при маленьком `C`.

## Как воспроизвести

1. Положить в корень проекта файлы:

```text
train.csv
test.csv
sample_submission.csv
```

2. Установить зависимости:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn catboost lightgbm xgboost
```

3. Запустить ноутбук:

```text
santander_customer_transaction_prediction.ipynb
```

4. Финальный файл для отправки находится здесь:

```text
result/catboost_strong_baseline.csv
```

## Основные файлы

- `santander_customer_transaction_prediction.ipynb` - основной ноутбук с EDA, обучением моделей и созданием submissions.
- `result/catboost_strong_baseline.csv` - лучший сохранённый submission.
- `image.png` - скриншот результата Kaggle для лучшего submission.
- `image-1.png` - скриншот результата для одного из baseline submissions.
