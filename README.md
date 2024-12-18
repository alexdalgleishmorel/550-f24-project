# SENG 550 FALL 2024 Group Project
### By Alex Dalgleish-Morel, Gabriel Cameron, Sandip Mishra, Chloe Bouchard

## Setup

> Prior to any setup steps, ensure you have docker installed and running on your machine.

From the root of the project, run:

```
docker compose up --build -d
```

To bash into the container:

```
docker exec -it python-app bash
```

Then, [download the training data](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) and add it the the `/data` directory as `raw.csv`.

Run this to setup the training, validation and test data:

```
python split_raw_data.py
```

Run this to train the model and have it predict on the test data:

```
python create_and_train_model.py
```

> Once the app is running, editing any code in `src/app` will automatically update in the container, and can be immediately executed.

## Teardown

To reset and re-create containers, run:

```
docker compose down
docker compose up --build -d
```
