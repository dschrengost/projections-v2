from prefect import flow, task
import time

@task
def ping():
    time.sleep(2)
    return "pong"

@flow(name="hello-prefect")
def hello():
    return ping()

if __name__ == "__main__":
    hello()