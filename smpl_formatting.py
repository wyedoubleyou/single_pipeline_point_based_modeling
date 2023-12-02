import pandas as pd 
import pickle
import pprint
import os 
import numpy as np
import torch
import subprocess as sp 

#array to vector, code from expose 
import torch 

def batch_rot2aa(Rs, epsilon=1e-7):
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);

        out_theta = std::acos(c);

        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """

    cos = 0.5 * (torch.einsum('bii->b', [Rs]) - 1)
    cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10 + epsilon)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

def batch_rodrigues(rot_vecs, epsilon=1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


#ori_ convert 3x3 array to 1x3 array (from expose func) (works)
import numpy as np


def trans_func_1():
    print("testing cross file func calling")

#transfer function 
# input: file name, ori_dir, expose_output_path, 
def trans_func(fname, ori_dir): 

    file = fname
    work_dir=ori_dir    

    input_path = f'{work_dir}/smplx/output/'
    pkl_name= input_path +file+'.pkl'


    #expose 
    img_format='.jpg'
    expose_path = f'{work_dir}/expose/samples/output/'          # MODIFY 

    folder = file + img_format + "_000"
    ori_pkl_path = expose_path+folder+"/"+folder+"_1.pkl"
    print("SMPL-X path: ", ori_pkl_path)

    df2 = pd.read_pickle(pkl_name)

    dict={}

    #camera_translation from expose file 
    smplx = pd.read_pickle(ori_pkl_path)
    transl=np.array([smplx['transl']])
    dict['camera_translation']=transl

    # global_orient
    go=df2['global_orient'][0]
    go = go.detach().cpu().numpy()
    gg1 = batch_rot2aa(torch.Tensor(go))
    newg=np.array(gg1)
    dict['global_orient']=newg

    #body pose 
    newb=[]
    bp=df2['body_pose'][0] #21 
    bp = bp.detach().cpu().numpy()

    for i in bp: 
        b=[i]
        b=np.array(b)
        new_angles=batch_rot2aa(torch.Tensor(b))
        new_angles=np.array(new_angles)
        newb.append(new_angles[0][0])
        newb.append(new_angles[0][1])
        newb.append(new_angles[0][2])

    newb=np.array([newb])
    newb=torch.FloatTensor(newb)
    dict['body_pose']=newb

    #camera_rotation (copy from ori files)
    # ref_pkl=f'{work_dir}/point_based_clothing/samples/internet_images/smpl/results/ori_itw_0.pkl'
    # ref = pd.read_pickle(ref_pkl)
    # dict['camera_rotation']=ref['camera_rotation']

    array = np.array([[[1., 0., 0.],
                  [1., 1., 0.],
                  [0., 0., 1.]]])
    dict['camera_rotation']=array


    #betas 
    betas=df2['betas']
    betas = betas.detach().cpu().numpy()
    dict['betas']=betas


    # vertices #10475  #not sure is internet_images requires vertices (this format follow psp)
    cr=df2['vertices']
    dict['vertices']=cr
    # print(cr)


    # save dict to pkl file 

    head, tail = os.path.split(pkl_name)
    out_path=input_path+'converted/'+tail

    with open(out_path, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("PKL file store in: ", out_path)

    review_loc= os.path.splitext(out_path)[0] + ".txt"

    with open(out_path, 'rb') as f1:
        b = pickle.load(f1)

    with open(review_loc, "w") as rf:
        pprint.pprint(b, stream=rf)

    print("TXT file store in: ",review_loc )


def graph_func(rgb_fname): 

    code = f'python exp/inference/inference.py  \
        --loadmodel data/pretrained_model/universal_trained.pth \
        --img_path ./img/{rgb_fname}.jpg \
        --output_path ./output/ \
        --output_name {rgb_fname}'
    test = sp.getoutput(code)
    print("dmeee")