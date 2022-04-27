import numpy as np
from LoadData import *
import numpy.matlib

class GetInst_A(object):
    def __init__(self, useful_sp_lab, img3d, gt, trpos, tepos):
        self.useful_sp_lab = useful_sp_lab
        self.img3d = img3d
        [self.r, self.c, self.l] = np.shape(img3d)
        self.num_classes = int(np.max(gt))
        self.img2d = np.reshape(img3d,[self.r*self.c, self.l])
        self.sp_num = np.array(np.max(self.useful_sp_lab), dtype='int')
        gt = np.array(gt, dtype='int')
        self.gt1d = np.reshape(gt, [self.r*self.c])
        self.gt_tr = np.array(np.zeros([self.r*self.c]), dtype='int')
        self.gt_te = self.gt1d
        self.trpos2d = np.array(trpos, dtype='int')
        self.trpos = (self.trpos2d[:,0]-1)*self.c+self.trpos2d[:,1]-1
        self.tepos2d = np.array(tepos, dtype='int')
        self.tepos = (self.tepos2d[:,0]-1)*self.c+self.tepos2d[:,1]-1
        self.tepos2d = self.tepos2d-1
        self.trpos2d = self.trpos2d-1


        
        self.sp_mean = np.zeros([self.sp_num, self.l])
        self.sp_center_px = np.zeros([self.sp_num, self.l]) 
        self.sp_label = np.zeros([self.sp_num]) 
        self.sp_label_sp = [] 
        self.ideal_sp_mat01 = np.zeros([self.sp_num, self.sp_num])
        self.trmask_sp = np.zeros([self.sp_num])
        self.temask_sp = np.ones([self.sp_num])
        self.sp_nei = [] 
        self.sp_nei_includeself = []
        self.sp_label_vec = []
        self.sp_A = [] 
        self.sp_A_notSym = []
        self.support = []
        self.px_nei_sp = []
        self.px_sp_01 = np.zeros([np.shape(self.trpos2d)[0]+np.shape(self.tepos2d)[0], self.sp_num])
        self.px_sp_A = np.zeros([np.shape(self.trpos2d)[0]+np.shape(self.tepos2d)[0], self.sp_num])
        
        self.Q = np.zeros([len(np.argwhere(gt>0)), self.sp_num])
        self.CalSpMean()    
        self.CalSpNei()
        self.CalSpA(scale = 1)
        self.ReprojectionQ()
        self.PxSpNei()
        
        
    def CalSpMean(self):
        self.gt_tr[self.trpos] = self.gt1d[self.trpos]
        mark_mat = np.zeros([self.r*self.c])
        mark_mat[self.trpos] = -1
        for sp_idx in range(1, self.sp_num+1): 
            region_pos_2d = np.argwhere(self.useful_sp_lab == sp_idx) 
            region_pos_1d = region_pos_2d[:, 0]*self.c + region_pos_2d[:, 1]
            px_num = np.shape(region_pos_2d)[0]
            if np.sum(mark_mat[region_pos_1d])<0:
                self.trmask_sp[sp_idx-1] = 1
                self.temask_sp[sp_idx-1] = 0
            region_fea = self.img2d[region_pos_1d, :]
            if self.trmask_sp[sp_idx-1] == 1:
                region_labels = self.gt_tr[region_pos_1d]
            else:
                region_labels = self.gt_te[region_pos_1d]
            self.sp_label[sp_idx-1] = np.argmax(np.delete(np.bincount(region_labels), 0))+1 
            region_pos_idx = np.argwhere(region_labels == self.sp_label[sp_idx-1])
            pos1 = region_pos_1d[region_pos_idx]
            self.sp_rps = np.mean(self.img2d[pos1, :], axis = 0)
            vj = np.sum(np.power(np.matlib.repmat(self.sp_rps, px_num, 1)-region_fea, 2), axis=1)
            vj= np.exp(-0.2*vj)
            self.sp_mean[sp_idx-1, :] = np.sum(np.reshape(vj, [np.size(vj), 1])*region_fea, axis=0)/np.sum(vj)
        self.sp_label_sp = self.sp_label
        x1 = np.array([i for i in range (self.sp_num)],dtype='int')
        self.ideal_sp_mat01[x1, np.array(self.sp_label_sp-1, dtype = 'int')] = 1
                            
                            
                            
        te_pos1 = np.argwhere(self.trmask_sp==0)
        self.ideal_sp_mat01[te_pos1, :] = 0
        self.ideal_sp_mat01[:, te_pos1] = 0
                    
                            
        
        
        sp_label_mat = np.zeros([self.sp_num, self.num_classes])
        for row_idx in range(np.shape(self.sp_label)[0]):
            col_idx = int(self.sp_label[row_idx])-1
            sp_label_mat[row_idx, col_idx] = 1
        self.sp_label_vec = self.sp_label
        self.sp_label = sp_label_mat
        self.sp_label_sp = self.sp_label
            
    def CalSpNei(self):
        for sp_idx in range(1, self.sp_num+1):
            nei_list = []
            region_pos_2d = np.argwhere(self.useful_sp_lab == sp_idx)
            r1 = np.min(region_pos_2d[:, 0])
            r2 = np.max(region_pos_2d[:, 0])
            c1 = np.min(region_pos_2d[:, 1])
            c2 = np.max(region_pos_2d[:, 1])
            for r in range(r1, r2+1):
                pos1 = np.argwhere(region_pos_2d[:, 0] == r)[:, 0]
                min_col = np.min(region_pos_2d[:, 1][pos1])
                max_col = np.max(region_pos_2d[:, 1][pos1])
                nc1 = min_col-1
                nc2 = max_col+1
                if nc1>=0:
                    nei_list.append(self.useful_sp_lab[r, nc1])
                    if r>0:
                        nei_list.append(self.useful_sp_lab[r-1, nc1])
                    if r<self.r-1:
                        nei_list.append(self.useful_sp_lab[r+1, nc1])
                if nc2<=self.c-1:
                    nei_list.append(self.useful_sp_lab[r, nc2])
                    if r>0:
                        nei_list.append(self.useful_sp_lab[r-1, nc2])
                    if r<self.r-1:
                        nei_list.append(self.useful_sp_lab[r+1, nc2])
            for c in range(c1, c2+1):
                pos1 = np.argwhere(region_pos_2d[:, 1] == c)[:, 0]
                min_row = np.min(region_pos_2d[:, 0][pos1])
                max_row = np.max(region_pos_2d[:, 0][pos1])  
                nr1 = min_row-1
                nr2 = max_row+1
                if nr1>=0:
                    nei_list.append(self.useful_sp_lab[nr1, c])
                if nr2<=self.r-1:
                    nei_list.append(self.useful_sp_lab[nr2, c])
            nei_list = list(set(nei_list))
            nei_list = [int(list_item) for list_item in nei_list]
            if 0 in nei_list:
                nei_list.remove(0)
            if sp_idx in nei_list:
                nei_list.remove(sp_idx)
            self.sp_nei.append(nei_list)
        self.sp_nei_includeself = self.sp_nei.copy()
        for sp_idx1 in range(1, self.sp_num+1):
            self.sp_nei_includeself[sp_idx1-1].append(sp_idx1)

    def CalSpA(self, scale = 1):
        sp_A_s1 = np.zeros([self.sp_num, self.sp_num])
        for sp_idx in range(1, self.sp_num+1):
            sp_idx0 = sp_idx-1 
            cen_sp = self.sp_mean[sp_idx0]
            nei_idx = self.sp_nei[sp_idx0] 
            nei_idx0 = np.array([list_item-1 for list_item in nei_idx], dtype=int)
            cen_nei = self.sp_mean[nei_idx0, :]
            dist1 = self.Eu_dist(cen_sp, cen_nei)
            sp_A_s1[sp_idx0, nei_idx0] = dist1
            
        self.sp_A.append(sp_A_s1)
        
        for scale_idx in range(0):              
            self.sp_A.append(self.AddConnection(self.sp_A[-1])) 
            del self.sp_A[0]     
           
        self.sp_A_notSym = self.sp_A.copy()
        self.sp_A_notSym[0] = (self.sp_A_notSym[0].T / np.sum(self.sp_A_notSym[0].T, axis = 0)).T
                                   
        
        for scale_idx in range(scale):  
            self.sp_A[scale_idx] = self.SymmetrizationMat(self.sp_A[scale_idx])
        for scale_idx in range(scale-1):  
            del self.sp_A[0]
            
            

    def AddConnection(self, A): 
        A1 = A.copy()
        num_rows = np.shape(A)[0]
        for row_idx in range(num_rows):
            pos1 = np.argwhere(A[row_idx, :]!=0) 
            for num_nei1 in range(np.size(pos1)):
                nei_ori = A[pos1[num_nei1, 0], :].copy() 
                pos2 = np.argwhere(nei_ori!=0)[:, 0] 
                nei1 = self.sp_mean[pos2, :]
                dist1 = self.Eu_dist(self.sp_mean[row_idx, :], nei1)
                A1[row_idx, pos2] = dist1
            A1[row_idx, row_idx] = 0
        return A1
             
    def AddConnectionFor01(self, A):
        A1 = A.copy()
        num_rows = np.shape(A)[0]
        for row_idx in range(num_rows):
            pos1 = np.argwhere(A[row_idx, :]!=0)
            for num_nei1 in range(np.size(pos1)):
                nei_ori = A[pos1[num_nei1, 0], :].copy() 
                pos2 = np.argwhere(nei_ori!=0)[:, 0] 
                nei1 = self.sp_mean[pos2, :]
                A1[row_idx, pos2] = 1
        return A1
         
             

    def SymmetrizationMat(self, mat):
        [r, c] = np.shape(mat)
        if r!=c:
            print('Input is not square matrix')
            return
        for rows in range(r):
            for cols in range(rows, c):
                e1 = mat[rows, cols]
                e2 = mat[cols, rows]
                if e1+e2!=0 and e1*e2 == 0:
                    mat[rows, cols] = e1+e2
                    mat[cols, rows] = e1+e2
        return mat
    def CalSupport(self, A, lam1):
        num1 = np.shape(A)[0]
        A_ = A+lam1*np.eye(num1)
        D_ = np.sum(A_, 1)
        D_05 = np.diag(D_**(-0.5))
        support = np.matmul(np.matmul(D_05, A_), D_05)
        return support
    
    def ReprojectionQ(self):
        num_all_px = len(self.trpos) + len(self.tepos)
        trtepos1d = np.concatenate((self.trpos, self.tepos), axis=0)
        trtepos2d = np.concatenate((self.trpos2d, self.tepos2d), axis=0)
        for px_idx0 in range(num_all_px):
            px_fea = self.img2d[trtepos1d[px_idx0]]
            sp_idx1 = self.useful_sp_lab[trtepos2d[px_idx0, 0], trtepos2d[px_idx0, 1]]
            neis = self.sp_nei[sp_idx1-1].copy()
            neis = [neis_elem-1 for neis_elem in neis]

            nei_sp_fea = self.sp_mean[neis]
            px_fea_repmat = np.matlib.repmat(px_fea, len(neis), 1)
            Q_row = np.exp(-0.2*np.sum(np.square(px_fea_repmat-nei_sp_fea), axis=1))
            self.Q[px_idx0, neis] = Q_row
        self.Q = self.Q/np.reshape(np.sum(self.Q, axis=1), [len(trtepos1d), 1])
        
        self.AllPxProcess()
    def AllPxProcess(self):
        num_all_px = len(self.trpos) + len(self.tepos)
        self.trmask = np.zeros([num_all_px])
        self.temask = np.ones([num_all_px])
        self.trmask[0:len(self.trpos)] = 1
        self.temask[len(self.trpos):num_all_px] = 1
        self.sp_label = np.zeros([num_all_px, self.num_classes])
        trtepos1d = np.concatenate((self.trpos, self.tepos), axis=0)
        class_idxes1 = self.gt1d[trtepos1d]
        self.sp_label[np.array([i1 for i1 in range(num_all_px)]), class_idxes1-1] = 1
    def PxSpNei(self):
        trtepos2d = np.concatenate((self.trpos2d, self.tepos2d), axis=0)
        px_num = np.shape(trtepos2d)[0]
        for px_idx in range(px_num):
            pos2d = trtepos2d[px_idx, :]
            sp_idx0 = self.useful_sp_lab[pos2d[0], pos2d[1]]-1
            self.px_nei_sp.append(self.sp_nei_includeself[sp_idx0])
            self.px_sp_01[px_idx, [elem-1 for elem in self.sp_nei_includeself[sp_idx0]]] = 1
            px_fea = self.img3d[pos2d[0], pos2d[1], :]
    def Eu_dist(self, vec, mat):
        rows = np.shape(mat)[0]
        mat1 = np.matlib.repmat(vec, rows, 1)
        dist1 = np.exp(-0.2*np.sum(np.power(mat1-mat, 2), axis = 1))
        return dist1   
    
    def CalLap(self, A):
        num1 = np.shape(A)[0]
        if A[0,0] == 1:
            A -= np.eye(num1)
        D_ = np.sum(A_, 1)
        D_05 = np.diag(D_**(-0.5))
        return np.eye(num1) - np.matmul(np.matmul(D_05, A), D_05)
    

        
     
        


