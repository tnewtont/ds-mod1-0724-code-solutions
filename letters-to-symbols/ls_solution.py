# This is actually an example of run-length encoding

from more_itertools import run_length as rl

def letters_to_symbols(s):

  # Creates a list of tuples, where each tuple is a letter-count pair
  counterList = list(rl.encode(s))

  # Swap: letter-count --> count-letter
  counterList_swapped = [(c[1], c[0]) for c in counterList]

  # Decomposes each count-letter pair into separate, individual elements of the
  # list (i.e. flatten out)
  # Reminder: each element of the list is a tuple, hence the ()
  counterList_flattened = list(sum(counterList_swapped,()))

  # Converts each count into string
  clf_str = list(map(str, counterList_flattened))

  # Concatenates each element of clf_str as a single string
  result = ''.join(map(str, counterList_flattened))

  print(result)


letters_to_symbols("AAAABBBCCDAAA")