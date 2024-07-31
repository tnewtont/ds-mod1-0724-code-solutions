# Count the number of open parentheses and the number of closed parentheses and
# make sure they are equal.

def BracketMatcher(s):
    countOpen = 0
    countClosed = 0

    for i in range(len(s)):
        if s[i] == '(':
            countOpen = countOpen + 1
        if s[i] == ')':
            countClosed = countClosed + 1
    if countOpen == countClosed:
        return True
    else: 
        return False


print(BracketMatcher('(a((kl(mns)t)uvwz)'))
