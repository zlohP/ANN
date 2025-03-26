def OR(x1,x2):
    w1,w2,theta = 0.1,0.1,0
    tmp = x1*w1 + x2*w2
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1

def NAND(x1,x2):
    w1,w2,theta = -0.5,-0.5,-0.7
    tmp = x1*w1 + x2*w2
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1
def AND(x1,x2):
    w1,w2,theta =0.5,0.5,0.7
    tmp = x1*w1 + x2*w2
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1

def XOR(x1,x2):
    s1=NAND(x1, x2)
    s2=OR(x1, x2)
    y=AND(s1,s2)
    return y

def adder(a1,a0,b1,b0):
    q0=XOR(a0,b0)
    s1=XOR(a1,b1)
    s2=AND(a0,b0)
    q1=XOR(s1,s2)
    q2=OR(AND(s1,s2),AND(a1,b1))
    print(q2,q1,q0)

adder(0,0,0,0)
adder(0,0,0,1)
adder(0,0,1,0)
adder(0,0,1,1)
adder(0,1,0,0)
adder(0,1,0,1)
adder(0,1,1,0)
adder(0,1,1,1)
adder(1,0,0,0)
adder(1,0,0,1)
adder(1,0,1,0)
adder(1,0,1,1)
adder(1,1,0,0)
adder(1,1,0,1)
adder(1,1,1,0)
adder(1,1,1,1)

