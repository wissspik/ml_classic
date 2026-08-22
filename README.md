# Customer Churn Prediction

Проект по бинарной классификации: предсказать, уйдёт ли клиент из сервиса (`Churn`).

Основная метрика — **ROC AUC**

## Цель проекта

Построить полный ML-пайплайн:

1. провести EDA;
2. обработать пропуски, категориальные признаки и выбросы;
3. сделать feature engineering;
4. обучить классические ML-модели;
5. обучить MLP;
6. сравнить модели через cross-validation;
7. сделать предсказания на test;
8. собрать итоговую таблицу результатов.

## Данные

Используются файлы:

- `train.csv` — обучающая выборка с целевой переменной `Churn`;
- `test.csv` — тестовая выборка без `Churn`;
- `sample_submission.csv` — пример файла для отправки предсказаний.

Целевая переменная:

- `Churn = Yes` — клиент ушёл;
- `Churn = No` — клиент остался.

## Что сделано в notebook

Основной notebook:

- `Customer_Churn.ipynb`

В нём есть следующие блоки:

1. **EDA**
   - размер и типы данных;
   - распределение `Churn`;
   - анализ пропусков;
   - проверка дубликатов;
   - распределения числовых признаков;
   - boxplot для анализа выбросов;
   - корреляционная матрица;
   - анализ категориальных признаков относительно `Churn`.

2. **Предобработка**
   - строки с пропущенным `Churn` удаляются;
   - числовые пропуски заполняются медианой train;
   - категориальные пропуски заполняются модой train;
   - `SeniorCitizen` обрабатывается как бинарный категориальный признак;
   - target приводится к формату `0/1`.

3. **Feature Engineering**
   - `X_base_no_features` — базовый набор признаков;
   - `FE_1_num_extra_services` — количество подключённых дополнительных сервисов;
   - `FE_2_charges_tenure` — признаки на основе `tenure`, `MonthlyCharges`, `TotalCharges`;
   - `FE_all_without_gender` — объединённый набор фичей без `gender`.

4. **Модели**
   - Logistic Regression baseline;
   - CatBoost;
   - LightGBM;
   - XGBoost;
   - MLP;
   - Stacking с финальной Logistic Regression.

5. **Оценка качества**
   - ROC AUC;
   - F1-score;
   - Accuracy;
   - время обучения;
   - итоговая таблица всех моделей.

6. **Предсказания на test**
   - сохраняются submission-файлы для лучших моделей.

## Feature Engineering

Используются три основных идеи:

### 1. Количество дополнительных сервисов

Создаётся признак `num_extra_services`, который считает, сколько дополнительных сервисов подключено у клиента:

- OnlineSecurity;
- OnlineBackup;
- DeviceProtection;
- TechSupport;
- StreamingTV;
- StreamingMovies.

Идея: клиент с большим количеством подключённых сервисов может быть сильнее привязан к продукту.

### 2. Признаки по платежам и сроку жизни клиента

Создаются признаки:

- `avg_monthly_charge`;
- `tenure_category`;
- `total_charges_category`.

Идея: отток часто связан с комбинацией длительности использования сервиса и размера платежей.

### 3. Удаление `gender`

По EDA `gender` слабо связан с `Churn`, поэтому отдельный feature set проверяет качество модели без этого признака.

## Модели и preprocessing

Для разных моделей используется разная обработка данных.

### Logistic Regression и MLP

Эти модели требуют числовую матрицу признаков, поэтому используется:

- `OneHotEncoder` для категориальных признаков;
- `StandardScaler` для числовых признаков.

### CatBoost

CatBoost получает категориальные признаки напрямую через `cat_features`.

### LightGBM и XGBoost

Категориальные признаки переводятся в pandas `category`.

## Cross-validation

Для основных моделей используется Stratified K-Fold CV:

- `Logistic Regression`: 5 folds;
- `CatBoost`: 5 folds;
- `LightGBM`: 5 folds в `RandomizedSearchCV`;
- `XGBoost`: 5 folds в `RandomizedSearchCV`;
- `MLP`: 5 folds, отдельно сравниваются activation `relu` и `tanh`.

Stratified CV используется, чтобы сохранить пропорции классов `Churn` в каждом fold.

## MLP

MLP обучается как отдельная нейросетевая модель через 5-fold CV.

Чтобы сравнить поведение сети, проверяются разные функции активации:

- `relu`;
- `tanh`.

В блоке MLP используются:

- лёгкая архитектура `(32,)`, чтобы 5-fold CV не был слишком долгим;
- `Adam` optimizer;
- L2-регуляризация через `alpha`;
- `early_stopping=True`;
- графики train loss по эпохам для каждого fold и activation;
- графики validation accuracy по эпохам для каждого fold и activation.

Эти графики нужны, чтобы увидеть переобучение: если train loss продолжает падать, а validation accuracy перестаёт расти или ухудшается, модель начинает overfit.

## Stacking

Используется stacking нескольких сильных моделей:

- CatBoost;
- LightGBM;
- XGBoost.

Их out-of-fold предсказания подаются в финальную Logistic Regression.

## Результаты

Финальная таблица результатов формируется в конце notebook и сортируется по `ROC AUC`.

В таблицу попадают:

- `model`;
- `feature_set`;
- `validation`;
- `roc_auc_mean`;
- `roc_auc_std`;
- `f1_mean`;
- `accuracy_mean`;
- `fit_time_sec`.

После полного запуска notebook лучшая модель выбирается автоматически по `roc_auc_mean`.

## Submission files

После запуска блока предсказаний создаются файлы в папке `result/`:

- `catboost_best_submission.csv`;
- `lightgbm_best_submission.csv`;
- `xgboost_best_submission.csv`;
- `stacking_logreg_submission.csv`.

## Как запустить проект

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Открыть notebook:

```bash
jupyter notebook Customer_Churn.ipynb
```

3. Запустить ячейки сверху вниз.

Важно: блоки CatBoost, LightGBM, XGBoost и Stacking могут обучаться долго, потому что используются 5-fold CV и подбор гиперпараметров.

## Структура проекта

```text
HM3/
├── Customer_Churn.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── requirements.txt
├── README.md
└── result/
    ├── catboost_best_submission.csv
    ├── lightgbm_best_submission.csv
    ├── xgboost_best_submission.csv
    └── stacking_logreg_submission.csv
```

## Основные выводы

- Для задачи Churn важнее всего оценивать качество по ROC AUC, а не только по Accuracy.
- `gender` почти не влияет на отток, поэтому его можно убрать без существенной потери качества.
- Признаки, связанные с контрактом, платежами, сроком жизни клиента и дополнительными сервисами, имеют больший смысл для предсказания оттока.
- Boosting-модели лучше подходят для табличных данных с категориальными признаками.
- Stacking может дать дополнительный прирост качества за счёт объединения нескольких сильных моделей.


## Анализ результата: почему лучшая модель — Stacking_LogReg

Ниже — объяснение по фактическому сохранённому прогону notebook и EDA-графикам.

### Фактический результат сохранённого прогона

В последней сохранённой итоговой таблице лучшей моделью стала `Stacking_LogReg` на наборе `FE_all_without_gender`.

| Модель | Feature set | ROC AUC |
|---|---|---:|
| Stacking_LogReg | FE_all_without_gender | 0.916450 |
| LightGBM | FE_all_without_gender | 0.916233 |
| XGBoost | FE_all_without_gender | 0.916224 |
| CatBoost | FE_all_without_gender | 0.915709 |

Разница между лучшими моделями небольшая: stacking лучше LightGBM примерно на `0.000217` ROC AUC и лучше XGBoost примерно на `0.000226`. Это не огромный разрыв, но для ROC AUC на большом датасете даже небольшой прирост может быть полезен.

### Что показали графики и EDA

Распределение target показывает дисбаланс классов: клиентов без оттока сильно больше, чем клиентов с `Churn = Yes`.

![Распределение Churn](figures/churn_distribution_train.png)

В train:

- `No`: 460377 клиентов;
- `Yes`: 133817 клиентов;
- доля churn примерно 22.5%.

Из-за этого ROC AUC выбран как главная метрика: accuracy может быть завышена из-за преобладания класса `No`, а ROC AUC оценивает качество ранжирования клиентов по вероятности ухода.

Графики распределений и boxplot показывают, что числовые признаки (`tenure`, `MonthlyCharges`, `TotalCharges`) имеют разные масштабы и нелинейные зависимости. Это объясняет, почему линейным моделям нужна нормализация, а tree-based/boosting модели хорошо подходят для такой задачи.

![Boxplot числовых признаков](figures/numeric_boxplots.png)

![Распределения числовых признаков](figures/numeric_distributions.png)

Корреляционная матрица полезна для первичной проверки числовых связей, но в этой задаче большая часть сильного сигнала находится не только в числовых, а в категориальных признаках.

![Корреляционная матрица](figures/correlation_matrix.png)

### Какие признаки сильнее всего объясняют churn

По таблицам EDA видно, что `gender` почти не влияет на churn:

- Female: 22.80% churn;
- Male: 22.23% churn.

Поэтому удаление `gender` в наборе `FE_all_without_gender` логично: признак слабый, а модель меньше цепляется за шум.

Сильные различия есть в других признаках:

| Признак | Наблюдение |
|---|---|
| `Contract` | Month-to-month: 42.05% churn, One year: 5.76%, Two year: 1.00% |
| `PaymentMethod` | Electronic check: 48.91% churn, остальные методы около 7-8% |
| `InternetService` | Fiber optic: 41.54% churn, DSL: 10.31%, No internet: 1.43% |
| `OnlineSecurity` | No: 40.61% churn, Yes: 8.68% |
| `TechSupport` | No: 40.16% churn, Yes: 9.65% |
| `SeniorCitizen` | 1: 50.03% churn, 0: 18.98% |
| `Dependents` | No: 29.14% churn, Yes: 7.28% |

Это объясняет, почему boosting-модели показывают высокий результат: они хорошо ловят нелинейные комбинации категориальных признаков, например связку `Contract + PaymentMethod + InternetService + tenure`.

### Почему помог feature engineering

Лучший feature set — `FE_all_without_gender`. В нём объединены две идеи:

1. `num_extra_services` — количество подключённых дополнительных сервисов.
2. Признаки из `tenure`, `MonthlyCharges`, `TotalCharges`:
   - `avg_monthly_charge`;
   - `tenure_category`;
   - `total_charges_category`.

Такие признаки добавляют модели более компактное описание поведения клиента. Например, клиент с коротким сроком жизни, высоким monthly charge, электронным чеком и контрактом month-to-month — типичный кандидат на churn.

### Почему Stacking_LogReg стал лучшим

CatBoost, LightGBM и XGBoost дали очень близкий ROC AUC: все находятся около `0.916`. Это значит, что каждая модель уже хорошо извлекает основной сигнал из данных, но делает немного разные ошибки.

Stacking работает за счёт этого различия:

1. CatBoost хорошо работает с категориальными признаками напрямую.
2. LightGBM быстро строит сильные деревья и хорошо ловит табличные зависимости.
3. XGBoost даёт похожую, но не идентичную модель ранжирования.
4. Финальная Logistic Regression получает out-of-fold вероятности этих моделей и учится комбинировать их.

В сохранённом output stacking видно, что базовые модели дают близкие, но не одинаковые вероятности для одних и тех же объектов. Например, для одного объекта вероятности были около `0.314`, `0.288`, `0.281`, а для другого высокого риска — `0.822`, `0.802`, `0.777`. Meta-model использует эти расхождения и немного улучшает итоговое ранжирование.

Итог: `Stacking_LogReg` стал лучшим не потому, что одна базовая модель резко сильнее остальных, а потому что он аккуратно объединил несколько сильных и похожих моделей. Прирост небольшой, но логичный.

