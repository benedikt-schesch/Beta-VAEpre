import pandas as pd


def text(s):
    s = str(round(s,2))
    if len(s.split(".")[1]) != 2:
        s = s+"0"*(2-len(s.split(".")[1]))
    return s

string = ""
#df = pd.read_csv("results/RESULT_48_32_student-por.csv",index_col=0)
df = pd.read_csv("results/RESULT__xAPI-Edu-Data.csv",index_col=0)
for i in df.iterrows():
    string += str(i[0])+" "
    for idx,j in enumerate(i[1]):
        m = float(j.split(" +/- ")[0])
        v = float(j.split(" +/- ")[1])
        m = m*100
        v = v*100
        string += "& $"+text(m)+" \pm "+text(v)+"$ "
    string += "\\\\ \n"
print(string)