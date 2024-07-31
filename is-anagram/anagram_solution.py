# Decompose each of the two strings as a sorted list of individual letters
# and then compare the two sorted lists (consider 'tar' vs 'art')

def is_anagram(str_1, str_2):
  if sorted(str_1) == sorted(str_2):
    return True
  else:
    return False

print(is_anagram('cautioned','education'))

print(is_anagram('cat', 'rat'))