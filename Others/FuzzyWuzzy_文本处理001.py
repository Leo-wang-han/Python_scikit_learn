'''
Fuzzywuzzy库是一个强大的模糊字符串匹配工具，
基于Levenshtein距离算法，可用于处理文本相似度匹配任务。
'''

from fuzzywuzzy import fuzz
# 1.相似度分析------中文字符串比较
str1 = "我爱北京天安门"
str2 = "我爱北京天安门广场"
# 计算相似度
similarity_score = fuzz.ratio(str1, str2)
print(f"相似度分数：{similarity_score}")

# 2.拼写检查
correct_words = ["长城","故宫","上海"]
use_input = input("请输入地名：")
best_match = max(correct_words, key=lambda word: fuzz.ratio(use_input, word))
print(f"推荐正确的词汇：{best_match}")

# 3.提示词
from fuzzywuzzy import process
search = "故宫的"
candidates = ["故宫博物院","故宫的历史","天安门"]
best_match = process.extract(search, candidates)
print(f"建议：{best_match[0]}")

# 4.数据清洗
data = ["apple", "aple", "banana", "bananna", "banana"]
cleaned_data = list(set(process.dedupe(data)))
print(cleaned_data)
