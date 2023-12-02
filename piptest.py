import os 
import subprocess 
import subprocess as sp
import sys 
from smpl_formatting import trans_func
from smpl_formatting import graph_func
from hello2 import inference_process 

# owd = os.getcwd()
mydir = os.getcwd() # original path
print("Original path: ", mydir)

mydir_new = os.chdir(mydir+"/expose") #rmb one time only, if not need restart 
newdir = os.getcwd()
print("Execute ExPose in:", newdir)

#checking for prevent duplicate expose 
proc = subprocess.Popen('pwd', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
o, e = proc.communicate()
pwd=o.decode('ascii').strip()
n=len(pwd.split('/'))
cont=False
if pwd.split('/')[n-1]=="expose" and pwd.split('/')[n-2]!="expose" : cont=True

if cont==False:
    print("Please restart the program")
    sys.exit()

# remove previous file in expose
newpwd=pwd.strip()+'/samples/test'
os.system(f'rm {newpwd}/*')

#add result/rgb to samples/test 
# rgb_file = 'Pip1'
rgb_file=sys.argv[1]

pb_sample = '/samples/internet_images/'   #TO NOTE: user can modify if change to another one 

# yw to note: modify mydir to result/rgb 
os.system(f"cp {mydir+pb_sample+'images/'+rgb_file+'.jpg'} {newpwd}")
print(f"Done copy {mydir+pb_sample+'images/'+rgb_file+'.jpg'} to {newpwd}") 


cmd='python demo.py --image-folder samples/test --exp-cfg data/conf.yaml --save-vis true --save-params true --save-mesh true --output-folder samples/output'
test = sp.getoutput(cmd)

ex_out= pwd+'/samples/output'
print(f"DONE generating SMPL-X parameter at {ex_out}")

# move ply to smplx folder 
ply_folder = f'{mydir}/smplx/ply_folder/'

ex_fname = rgb_file+".jpg_000"
newf= ex_fname.split('.')[0]+".ply"
ply_name = ex_out+"/"+ex_fname+'/'+ex_fname+'.ply'
final_ply=ply_folder+newf

command = sp.getoutput(f'cp {ply_name} {final_ply}')
print(f"DONE cp {ply_name} {final_ply}")


# work on smplx folder 

mydir_new = os.chdir(mydir+"/smplx") #rmb one time only, if not need restart 
newdir = os.getcwd()
print("\nExecute SMPL-X in:", newdir)

proc = subprocess.Popen('pwd', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
o, e = proc.communicate()
pwd=o.decode('ascii').strip()
n=len(pwd.split('/'))
cont=False
if pwd.split('/')[n-1]=="smplx" and pwd.split('/')[n-2]!="smplx" : cont=True

if cont==False:
    print("Please restart the program")
    sys.exit()

# TO NOTE: if all use the same library
smplx_cmd ='python -m transfer_model --exp-cfg config_files/smplx2smpl.yaml'
cmd = sp.getoutput(smplx_cmd)

# due library issue i use another env (py38) for smplx
# subprocess.run(f'conda run -n py38 python {mydir+"/"}piptest2.py', shell=True)

print("DONE converted SMPL-X to SMPL")
print("Formating SMPL...") 

test=trans_func(rgb_file,mydir)

smpl_path=mydir+"/smplx/output/converted/"
smpl_out = f"{smpl_path+rgb_file+'.pkl'}"
smpl_check =os.path.isfile(smpl_out)

if smpl_check==True: 
    # sys conttinue 
    print("DONE Formating SMPL") 
else:
    print(test)
    print("\nSMPL not generated")
    sys.exit()


# Graphonomy: generating clothing segmentation masks 

mydir_new = os.chdir(mydir+"/Graphonomy") #rmb one time only, if not need restart 
newdir = os.getcwd()
print("\nExecute Graphonomy in:", newdir)


os.system(f"cp {mydir+pb_sample+'images/'+rgb_file+'.jpg'} {newdir+'/img/'}")
print(f"Done copy {mydir+pb_sample+'images/'+rgb_file+'.jpg'} to {newdir+'/img/'}") 

code = f'python exp/inference/inference.py  \
        --loadmodel data/pretrained_model/universal_trained.pth \
        --img_path ./img/{rgb_file}.jpg \
        --output_path ./output \
        --output_name /{rgb_file}'

test = sp.getoutput(code)

grap_out = f"{mydir+'/Graphonomy/output/'+rgb_file+'.png'}"
grap_check =os.path.isfile(grap_out)

if grap_check==True: 
    # sys conttinue 
    print(f"Clothing segmentation mask generated at {grap_out}")
else:
    print(test)
    print("\nClothing Segmentation Mask not generated")
    sys.exit()

#is combine all the directory this no need
mydir_new = os.chdir(mydir)
newdir = os.getcwd()
print("\nCurrent directory is", newdir)

grap_s='segmentations'
smpl_s='smpl'

grap_s_path= newdir+pb_sample+grap_s
smpl_s_path= newdir+pb_sample+smpl_s

os.system(f"cp {grap_out} {grap_s_path}")
print(f"Done copy {grap_out} to {grap_s_path}") 

os.system(f"cp {smpl_out} {smpl_s_path}")
print(f"Done copy {smpl_out} to {smpl_s_path}") 


inference_process(rgb_file)
