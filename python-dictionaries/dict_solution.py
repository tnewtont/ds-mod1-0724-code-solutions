def deci_to_bi(n):
  deci_bi_dict = {} # Initiate dictionary

  for num in range(n): 
    bi_value = "" # Initiate binary result (values of dictionary)
    temporary = num

    while temporary != 0:
      remainder = temporary % 2
      bi_value = str(remainder) + bi_value
      temporary //= 2 # Floor

    if len(bi_value) < 4: # If the binary result is less than 4 chars long
        bi_value = '0'*(4-len(bi_value)) + bi_value

    deci_bi_dict[num] = bi_value # Key to value

  return deci_bi_dict

print(deci_to_bi(16))