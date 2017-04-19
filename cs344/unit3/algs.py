def ceil(n, k):
  return n / k + (n % k > 0)

def scan(arr):
  n = len(arr)
  ret = [a for a in arr]
  inc = 2
  while True:
    for i in range(1, ceil(n, inc) + 1):
      idxLeft, idxRight = i * inc - 1 - inc / 2, i * inc - 1
      if idxRight>= n:
        continue
      ret[idxRight] += ret[idxLeft]
    if ceil(n, inc) == 1:
      break
    inc *= 2

  inc /= 2
  while True:
    for i in range(1, ceil(n, inc) + 1):
      idxLeft, idxRight = i * inc - 1, i * inc + inc / 2 - 1
      if idxRight>= n:
        continue
      ret[idxRight] += ret[idxLeft]
    if inc == 2:
      break
    inc /= 2
  return ret

for n in [8, 7, 6, 5]:
  arr = range(n)
  print scan(arr)
