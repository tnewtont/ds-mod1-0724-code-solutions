L = 'abcdefghijklmnopqrstuvwxyz'

def create_two(input):
  # Create a list of numbers corresponding to the letters
    L1 = []
    for j in range(len(input)):
      L1.append(j + 1)
    print(L1)

  # Create a list of individual letters
    L2 = []
    L2 = [i for i in input]
    print(L2)

create_two(L)

def combine_two(input):
  enumerated_letters = enumerate(input, start=1)
  print(list(enumerated_letters))

combine_two(L)


