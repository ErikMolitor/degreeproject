import random


y= []
nr_namn = input("Hur många namn vill du skriva in? ")
nr_namn = int(nr_namn)
for i in range(nr_namn):
    x = input("Skriv in ett namn: ")
    y.append(x)

print(y)
random.shuffle(y)
idx = round(len(y)/2)
y1 = y[:idx]
y2 = y[idx:]


while len(y1) or len(y2) > 0:
    name1 = y1[0]
    name2 = y2[0]
    y1 = y1[1:]
    y2 = y2[1:]

    if len(y1) == 0 and len(y2) == 1:
        print(name1 +" ska leka med " + name2 + " och " + y2[0])
        y2 = y2[1:]
    
    elif len(y1) == 1 and len(y2) == 0:
        print(name1 +" ska leka med " + name2 + " och " + y1[0])
        y1 = y1[1:]

    else:
        print(name1 +" ska leka med " + name2 ) 

