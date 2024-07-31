L = 'abcdefghijklmnopqrstuvwxyz'

def create_combine_two(input):
  # Create a list of numbers corresponding to the letters
    L1 = []
    for letter in range(len(input)):
      L1.append(letter + 1)

  # Create a list of individual letters
    L2 = []
    L2 = [i for i in input]

  # Combine the two lists
    combined_list = []

    for l in range(len(input)):
      combined_list.append((L1[l],L2[l]))

    print(combined_list)

create_combine_two(L)

