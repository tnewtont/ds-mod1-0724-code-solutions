def deci_to_bi(number):

  # List of keys (decimals)
  deci_list = []
  for i in range(number):
    deci_list.append(i)

  # List of values (binaries)
  bi_list = []
  for i in range(number):
    bi_list.append("{:04b}".format(i))

  # Creates dictionary
  deci_bi_dict = dict(zip(deci_list, bi_list))
  print(deci_bi_dict)

deci_to_bi(16) # 0 to 15