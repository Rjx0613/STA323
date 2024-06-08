import time
import json
from kafka import KafkaProducer
import pandas as pd

# Kafka 服务的地址和主题
broker = 'localhost:64046'
topic = 'qa'

# 创建 Kafka 生产者
producer = KafkaProducer(bootstrap_servers=broker, value_serializer=lambda m: json.dumps(m).encode('utf-8'))
df = pd.read_csv('./test.csv')

print(">>> Now successfully start the kafka producer process. Run in background. Do not exit...")

for _, row in df.iterrows():
    message = {"Input": row["Input"], "Output":row["Output"]}
    producer.send(topic, message)
    time.sleep(0.5)
    
# 确保所有消息都被发送
producer.flush()