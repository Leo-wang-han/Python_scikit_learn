#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Date: 2024/6/14 0014
@FILE: 链表与数组04
@Author: Leo
"""
import numpy as np
import random
"""
4.1数组（array）
    数组是一种线性数据结构，其将相同类型的元素存储在连续的内存空间中。我们将
    元素在数组中的位置成为钙元素的索引（index）
"""

# 1.初始化数组---无初始值、给定初始值
arr = np.array([0]*5)
num = np.array([1,2,3,4,5])
print(arr)

# 2.访问元素
def random_access(nums: list[int]) -> int:
    random_index = random.randint(0, len(num)-1)    # 随机抽取num数组的一个索引
    random_num = num[random_index]    # 根据索引访问元素，时间O(1)
    return random_num


# 3.插入元素
# 数组元素的内存是紧挨着的，中间插入一个元素，由于数组长度是固定的，会导致尾部元素的丢失
def inset(nums: list[int], number: int, index: int):
    for i in range(len(nums)-1, index, -1):
        nums[i] = nums[i-1]    # 把索引index以及之后的所有元素往后移动一位
    nums[index] = number     # 插入number到nums索引=index处
    return nums

# 4.删除元素
def delete(nums: list[int], index: int):
    # 删除索引index处的元素
    # 把索引后的所有元素向前
    for i in range(index, len(nums)):
        nums[i] = nums[i+1]
    return nums

# 5.查找元素---遍历数组
# 6.扩容数组，将原数组元素依次复制到新数组---O(n)
# 7.优缺点，应用
"""
优点：
空间效率高，支持随机访问，缓存局部性
缺点：
插入与删除效率低，长度不可变，空间浪费
典型应用：
随机访问、排序和搜索、查找表、机器学习
"""


"""
4.2链表(ListNode)
链表是一种线性数据结构，其中的每个元素都是一个节点对象，各个节点通过"应用"相连接。
应用记录了下一个节点的内存地址，通过他可以从当前节点访问到下一个节点。内存地址无需连续。
节点（ListNode）:值+引用（指针）,链表比数组占用更多的内存空间
ListNode.next依次访问节点
"""
# 1.初始化链表
# n0 = ListNode(1)
# n1 = ListNode(2)
# n0.next = n1

# 2.插入节点
# 改变两个节点引用（指针），时间复杂度O(1)
# def insert_ListNode(n0: ListNode,P: ListNode):
#     n1 = n0.next
#     P.next = n1
#     n0.next = P

# 3.删除节点
# 只需改变一个节点的引用即可

# 4.访问节点
# 程序需要从头部节点出发，逐个向后遍历，直到找到目标节点，时间复杂度O(0)

# 5.常见链表类型
# 单项链表：节点包含值和指向下一节点的引用两项数据
# 环形链表：单项链表的尾节点指向头结点
# 双向链表：双向链表的节点定义同事包含指向后继结点（下一个节点）和前驱节点（前一个节点）的引用

# 6.链表典型应用
# 单向链表：栈与队列、哈希表、图
# 双向链表：红黑树、B树、浏览器历史、LRU（缓存淘汰）算法、
# 环形链表：时间片轮转调度算法、数据缓冲区



"""
4.3列表(list)
是一个抽象的数据结构概念，它表示元素的有序集合、支持元素访问、修改、添加、删除和遍历等操作。
"""
# 1.初始化
list = []
list = [1, 2, 3]

# 1.访问元素
num = list[1]   # 访问索引1出的元素，时间O(1)
list[1] = 0    # 将索引1处的元素更新为0

# 2.插入与删除
list.clear()    # 清空列表
list.append(1)    # 在尾部添加元素1
list.insert(3,6)    # 在索引3处插入数字6
list.pop(3)    # 删除索引3处的元素，默认尾部

# 3.遍历、拼接、排序
# 索引遍历元素
for i in range(len(list)):
    print(list[i])
# 直接遍历元素
for num in list:
    print(num)
# 同时遍历索引和值
for index,num in enumerate(list):
    print({index: num})

# 排序
list.sort()

# 拼接
list2 = [2,1]
list += list2


#内存与缓存
"""
硬盘（hard disk0）:硬盘用于长期存储大量数据
内存（random-access memory,RAM）：内存用于临时存储程序运行中正在处理的数据
缓存（cache memory）：缓存用量存储经常访问的数据和指令；通过缓存行、预取机制以及空间局部性和时间局部性等数据加载机制，为
CPU提供更快速数据访问
"""







