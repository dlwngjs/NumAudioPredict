import random

# 0~9 중에서 랜덤으로 5개 숫자 뽑기
random_labels = random.sample(range(10), 5)
random_file = random.sample(range(8), 5)

print(random_labels)
print(random_file)