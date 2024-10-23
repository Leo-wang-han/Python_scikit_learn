import threading

# 1.线程的初步认识
# 想象你是个厨房大厨，一边炒菜一边洗菜，这就是多线程的日常。
# 在Python里，threading模块就是我们的厨房神器。


def wash():
    print("洗菜中...")


def cook():
    print("炒菜中...")


# 创建线程对象
# target参数指定线程要执行的函数
thread1 = threading.Thread(target=wash)
thread2 = threading.Thread(target=cook)

# 启动线程   start()让线程开始执行
thread1.start()
thread2.start()

# 等待所有线程完成    join()确保主线程等待这些小线程们完成它们的任务
thread1.join(timeout=1)
thread2.join(timeout=1)

print("菜做好了")

# 2.线程同步
# 在多线程世界，如果两个线程同时操作同一资源（比如共享食材），
# 就可能出乱子。这时就需要“锁”来帮忙了，Python里的锁叫Lock。
shared_resource = 0
lock = threading.Lock()


def increase():
    global shared_resource
    # 上锁，防止同时访问
    lock.acquire()
    shared_resource += 1
    # 解锁，释放控制权
    lock.release()


threads = [threading.Thread(target=increase) for i in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("共享资源的最终值：",shared_resource)


# 3.死锁
# 但锁用不好也会出问题，
# 就像两个厨师互相等待对方手中的锅，形成了死锁。要小心设计，避免循环等待。


# 4. 线程池
# 想象一下，如果你每次炒菜都要新雇一个厨师，那得多浪费？
# 线程池(ThreadPoolExecutor)就是解决这个问题的神器，它预先创建好一些线程，重复利用

from concurrent.futures import ThreadPoolExecutor


def task(n):
    print(f"执行任务{n}")


# ThreadPoolExecutor创建了一个最多有5个线程的池
with ThreadPoolExecutor(max_workers=5) as executor:
    # map()函数并行执行任务列表中的每个任务。
    executor.map(task, range(1, 6))


# 5. 守护线程
# 护线程就像厨房的清洁工，在所有其他线程完成后默默清理。
# 通过setDaemon(True)设置线程为守护线程。

def clearner():
    while True:        # 假设这是一个无限循环，清理任务
        print("打扫厨房"+"\n")
        if not thread2:      # 假定函数检查其他线程是否还在运行
            break


clean_thread = threading.Thread(target=clearner)
clean_thread.setDaemon(True)
clean_thread.start()
# 其他线程的代码...
print("厨房关闭，清洁完成")


# 6. 线程优先级
# 虽然Python标准库没有直接提供线程优先级的功能，但可以通过队列等间接实现。
# 不过，大多数情况下，Python的线程调度是公平的，不需要担心。


# 7. 全局解释器锁（GIL）
# Python的GIL是一个让人又爱又恨的东西，它保证了任何时刻只有一个线程在执行Python字节码，
# 这对多核CPU来说不是个好消息。但在I/O密集型任务中，GIL的影响没那么大

# 8. 线程局部存储
# 不同线程需要不同的“调料”怎么办？threading.local()来帮忙，
# 它提供了线程本地的存储空间。


# import threading
# local_data = threading.local()
#
#
# def set_data():
#     local_data.value = "这是我的调料"
#
#
# def get_data():
#     print(local_data.value)
#
#
# t1 = threading.Thread(target=set_data)
# t2 = threading.Thread(target=get_data)
#
# t1.start()
# t2.start()
#
# t1.join()
# t2.join()


# 9. 线程异常处理
# 线程中的异常不会自动传递到主线程，需要用try-except捕获处理。
def risk_task():
    raise ValueError("出错啦")

try:
    t = threading.Thread(target=risk_task)
    t.start()
    t.join()
except ValueError as e:
    print(f"捕获到异常{e}")





