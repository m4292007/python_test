a_list = [3, 4, 62, 27, 83, 956, 26, 58, 3, 78, 168, 64, 78]

def print_list(a):
    for i in a:
        print(i)

print_list(a_list)

def countdown(n):
    if n == 0:
        print("Blastoff!")
    else:
        print(n)
        countdown(n-1)


countdown(2)


def multi(a):
    i = 1
    while i <=9:
        print (a, ' * ', i, ' = ', a * i)
        i = i + 1

multi(3)        
