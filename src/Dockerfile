FROM python:3.9

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["tail", "-f", "/dev/null"]
