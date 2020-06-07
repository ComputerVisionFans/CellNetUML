

import time

def readlsf():
    #df=pd.read_table(lsf,header=None)
    #df = pd.read_csv(lsf, sep=" ", header=None)
    tm= time.time()

    lsffile = open("ghostresnet with cellyolo 80k.txt", "r")
    with open(("filtered_acc.txt"), "w") as file:
        for f in lsffile.readlines():
            if f.find('* Acc@')>0:
                print(f)
                file.write(f)
        else:
            pass
readlsf()