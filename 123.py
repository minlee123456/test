arr = []
N = int(input("N: "))
for I in range(N):
    min_map = input(":")
    arr.append(list(map(int,min_map.split())))

def counta(rl,ud):
    count=0
for r in range(rl-1,rl+1):
        for d in range(ud-1,ud+1):
            count += 1

for i in range(N):
    for j in range(N):
        if arr[i][j] == 1:
            print("*")
        else:
            print(counta(i,j))
    print()