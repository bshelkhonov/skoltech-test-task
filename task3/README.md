# Запуск

```bash
docker build -t ubuntu_image .
docker run -itd --name ubuntu_container ubuntu_image
docker exec ubuntu_container bash ./scripts/frequency.sh ./texts/dracula.txt
docker exec ubuntu_container bash ./scripts/most_frequent_words.sh ./texts/dracula.txt results
```
