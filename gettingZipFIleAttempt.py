#NECESSARY MODULES
import zipfile
import os     
import shutil                  

#LOCATION TO DR. GRIEGG'S ZIP FILES
print(os.listdir('/mnt/network/sgrieggs/rsna'))
zip_path = '/mnt/network/sgrieggs/rsna/rsna-breast-cancer-512-pngs.zip'


# #CHECKS IF PATH EXISTS
if not os.path.exists(zip_path):
    print('1 ERROR: file does not exist')
    
# #PUTS THE ZIP FILE FROM PATH INTO VARIABLE zip_file
try:
    zip_file = zipfile.ZipFile(zip_path, 'r')
except zipfile.BadZipFile: #CHECKS IF zip file is bad file
    print('2 Error: bad file')
except Exception as e: #CHECKS FOR ALTERNATE EXCEPTIONS
    print('3 Error ', e)
    
#CHECKING IF SUCCESSFUL, PRINTS ERROR IF NOT
if zip_file:
    #zip_file.printdir() #PRINTS CONTENTS OF ZIP FILE   
    zip_file.extractall('/home/zgxdc/USOAR/train_images') #Should extract files to that directory
    zip_file.close()
else:
    print('FAILED')
    
csvPath = '/mnt/network/sgrieggs/rsna/val_split.csv'
exportPath = '/home/zgxdc/USOAR'
shutil.copy(csvPath, exportPath)