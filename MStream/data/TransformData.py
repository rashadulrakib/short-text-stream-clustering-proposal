import re

file1=open("Biomedical","r")
lines = file1.readlines()
file1.close()

file1=open("Biomedical-mstream","w")

count=1 
for line in lines:
  line = line.strip()
  if len(line)==0:
    continue
  arr = re.split("\t", line)
  true=arr[0].strip()
  text=arr[1].strip()
  
  if len(true)==0 or len(text)==0:
    continue
  #{"Id": "000001", "clusterNo": 65, "textCleaned": "centrepoint winter white gala london"}  
  file1.write('{"Id": "'+str(count).zfill(6)+'", "clusterNo": '+str(true)+', "textCleaned":"'+text+'"}\n')
  count=count+1  
  
file1.close()  
    