# 多线程实战，下载图片
import requests
# from PIL import Image
# from io import BytesIO
# from tqdm import tqdm
import threading
from queue import Queue


def download_img(url):
   response = requests.get(url)
   with open(f"../files_save/image_{url[-5:]}.jpg",'wb') as f:
    f.write(response.content)
    print(f"图片下载完成{url}")


# 假设的url
image_urls = ["https://img1.baidu.com/it/u=558897944,3706361995&fm=253&fmt=auto&app=120&f=JPEG?w=1202&h=800",
              "https://img1.baidu.com/it/u=185940082,852979514&fm=253&fmt=auto&app=138&f=JPEG?w=903&h=500"]
queue = Queue()    # FIFO
threads = []

for url in image_urls:
    queue.put(url)


def worker():
    while not queue.empty():
        url = queue.get()
        download_img(url)
        queue.task_done()


for _ in range(3):   # 启动3个下载线程
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# 等待所有下载任务完成
for t in threads:
    t.join()

print("所有图片下载完成")
