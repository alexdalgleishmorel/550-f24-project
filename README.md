# SENG 550 FALL 2024 Group Project
### By Alex Dalgleish-Morel, Gabriel Cameron, Sandip Mishra, Chloe Bouchard

## Setup

> Prior to any setup steps, ensure you have docker installed and running on your machine.

From the root of the project, run:

```
docker compose up --build -d
```

If this is your first time setting things up, wait for the database logs to finish initializing:

```
docker logs -f nyc_taxi_data_db
```

Once the setup is complete, bash into the container:

```
docker exec -it python-app bash
```

Then you can run any python files here in the intial directory, for example:

```
python app.py
```

> Once the app is running, editing any code in `app/src` will automatically update in the container, and can be immediately executed.

## Teardown

To reset the database and containers, run:

```
docker compose down
docker volume rm 550-f24-project_db_data
docker compose up --build -d
```
