def openDoll(Doll):
    if (Doll == 1):
        print("Opening Dolls complete")
    else:
        openDoll(Doll - 1)
        print(openDoll(1))
def firstMethod():
    print("Iam the first method")
    secondMethod()
def secondMethod():
    print("I am the second method")
    thirdMethod()

def thirdMethod():
    print("Iam the third method")
    fourthMethod()

def fourthMethod():
    print("I am the fourth and the last method!!")
    firstMethod()

print(firstMethod())
