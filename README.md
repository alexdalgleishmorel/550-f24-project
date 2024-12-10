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

Then you can run any python files here in the intial directory, for example:

***For PySpark to work on the container, you need to run:***
```
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
export PATH=$JAVA_HOME/bin:$PATH
```

> This above command may differ on your machine, just check what's in `/usr/lib/jvm/` on the container.

```
python app.py
```

> Once the app is running, editing any code in `src/app` will automatically update in the container, and can be immediately executed.

## Teardown

To reset and re-create containers, run:

```
docker compose down
docker compose up --build -d
```
