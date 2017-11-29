import numpy as np
import random as rd
import copy as cp
def pus(lisp):
    lis=cp.deepcopy(lisp)
    try:
        lis.remove(0)
    except ValueError:
        j=1
    try:
        lis.remove(0)
    except ValueError:
        j=1
    try:
        lis.remove(0)
    except ValueError:
        j=1
    try:
        lis.remove(0)
    except ValueError:
        j=1
    while len(lis)<4:
        lis.append(0)
    return(lis)
def cpq(lis):
    if pus(lis)==lis:
        [lisp,v]=hb(lis)
        if lisp==lis:
            return(1)
        return(0)
    return(0)
def hb(lisp):
    lis=cp.deepcopy(lisp)
    v=0
    if lis[0]==lis[1]:
        lis[0]=lis[0]*2
        lis[1]=0
        v+=lis[0]
    if lis[1]==lis[2]:
        lis[1]=lis[2]*2
        lis[2]=0
        v+=lis[1]
    if lis[2]==lis[3]:
        lis[2]=lis[3]*2
        lis[3]=0
        v+=lis[2]
    lis=pus(lis)
    return([lis,v])
    
            
class State:
    def __init__(self,arrawhe=0):
        if arrawhe==0:
            self.a=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            self.a=arrawhe
    def upusa(self):
        arra=self.a
        return(cpq([arra[0],arra[4],arra[8],arra[12]])*
               cpq([arra[1],arra[5],arra[9],arra[13]])*
               cpq([arra[2],arra[6],arra[10],arra[14]])*
               cpq([arra[3],arra[7],arra[11],arra[15]]))
    def dpusa(self):
        arra=self.a
        return(cpq([arra[12],arra[8],arra[4],arra[0]])*
               cpq([arra[13],arra[9],arra[5],arra[1]])*
               cpq([arra[14],arra[10],arra[6],arra[2]])*
               cpq([arra[15],arra[11],arra[7],arra[3]]))
    def lpusa(self):
        arra=self.a
        return(cpq([arra[0],arra[1],arra[2],arra[3]])*
               cpq([arra[4],arra[5],arra[6],arra[7]])*
               cpq([arra[8],arra[9],arra[10],arra[11]])*
               cpq([arra[12],arra[13],arra[14],arra[15]]))
    def rpusa(self):
        arra=self.a
        return(cpq([arra[3],arra[2],arra[1],arra[0]])*
               cpq([arra[7],arra[6],arra[5],arra[4]])*
               cpq([arra[11],arra[10],arra[9],arra[8]])*
               cpq([arra[15],arra[14],arra[13],arra[12]]))
    def avai(self):
        return([self.upusa(),self.dpusa(),self.lpusa(),self.rpusa()])
    def rnew(self,needdata=0):
        arra=self.a
        i=0
        while i==0:
            j=rd.randint(0,15)
            if arra[j]==0:
                i=1
                k=2*rd.randint(1,2)
                new=self.ad(j,k)
                #new.prt()
                if needdata==1:
                    #new.prt()
                    return([new,j,k])
                
                
                return(new)
    def ad(self,j,n):
        new=cp.deepcopy(self)
        new.a[j]=n
        #new.prt()
        return(new)
    def upus(self):
        if self.upusa()==0:
            arra=self.a
            v=0
            [arra[0],arra[4],arra[8],arra[12]]=pus([arra[0],arra[4],arra[8],arra[12]])
            [arra[1],arra[5],arra[9],arra[13]]=pus([arra[1],arra[5],arra[9],arra[13]])
            [arra[2],arra[6],arra[10],arra[14]]=pus([arra[2],arra[6],arra[10],arra[14]])
            [arra[3],arra[7],arra[11],arra[15]]=pus([arra[3],arra[7],arra[11],arra[15]])
            [[arra[0],arra[4],arra[8],arra[12]],v1]=hb([arra[0],arra[4],arra[8],arra[12]])
            [[arra[1],arra[5],arra[9],arra[13]],v2]=hb([arra[1],arra[5],arra[9],arra[13]])
            [[arra[2],arra[6],arra[10],arra[14]],v3]=hb([arra[2],arra[6],arra[10],arra[14]])
            [[arra[3],arra[7],arra[11],arra[15]],v4]=hb([arra[3],arra[7],arra[11],arra[15]])
            v+=v1+v2+v3+v4
            #self.a=arra
            self.rnew()
            return(v)
    def dpus(self):
        if self.dpusa()==0:
            arra=self.a
            v=0
            [arra[12],arra[8],arra[4],arra[0]]=pus([arra[12],arra[8],arra[4],arra[0]])
            [arra[13],arra[9],arra[5],arra[1]]=pus([arra[13],arra[9],arra[5],arra[1]])
            [arra[14],arra[10],arra[6],arra[2]]=pus([arra[14],arra[10],arra[6],arra[2]])
            [arra[15],arra[11],arra[7],arra[3]]=pus([arra[15],arra[11],arra[7],arra[3]])
            [[arra[12],arra[8],arra[4],arra[0]],v1]=hb([arra[12],arra[8],arra[4],arra[0]])
            [[arra[13],arra[9],arra[5],arra[1]],v2]=hb([arra[13],arra[9],arra[5],arra[1]])
            [[arra[14],arra[10],arra[6],arra[2]],v3]=hb([arra[14],arra[10],arra[6],arra[2]])
            [[arra[15],arra[11],arra[7],arra[3]],v4]=hb([arra[15],arra[11],arra[7],arra[3]])
            v+=v1+v2+v3+v4
            self.a=arra
            #self.rnew()
            return(v)
    def lpus(self,ifprt=0):
        if self.lpusa()==0:
            arra=self.a
            v=0
            [arra[0],arra[1],arra[2],arra[3]]=pus([arra[0],arra[1],arra[2],arra[3]])
            [arra[4],arra[5],arra[6],arra[7]]=pus([arra[4],arra[5],arra[6],arra[7]])
            [arra[8],arra[9],arra[10],arra[11]]=pus([arra[8],arra[9],arra[10],arra[11]])
            [arra[12],arra[13],arra[14],arra[15]]=pus([arra[12],arra[13],arra[14],arra[15]])
            [[arra[0],arra[1],arra[2],arra[3]],v1]=hb([arra[0],arra[1],arra[2],arra[3]])
            [[arra[4],arra[5],arra[6],arra[7]],v2]=hb([arra[4],arra[5],arra[6],arra[7]])
            [[arra[8],arra[9],arra[10],arra[11]],v3]=hb([arra[8],arra[9],arra[10],arra[11]])
            [[arra[12],arra[13],arra[14],arra[15]],v4]=hb([arra[12],arra[13],arra[14],arra[15]])
            v+=v1+v2+v3+v4
            self.a=arra
            if ifprt==1:
                print('1')
                self.prt()
            #self.rnew()
            return(v)
    def rpus(self,ifprt=0):
        if self.rpusa()==0:
            arra=self.a
            v=0
            [arra[3],arra[2],arra[1],arra[0]]=pus([arra[3],arra[2],arra[1],arra[0]])
            [arra[7],arra[6],arra[5],arra[4]]=pus([arra[7],arra[6],arra[5],arra[4]])
            [arra[11],arra[10],arra[9],arra[8]]=pus([arra[11],arra[10],arra[9],arra[8]])
            [arra[15],arra[14],arra[13],arra[12]]=pus([arra[15],arra[14],arra[13],arra[12]])
            [[arra[3],arra[2],arra[1],arra[0]],v1]=hb([arra[3],arra[2],arra[1],arra[0]])
            [[arra[7],arra[6],arra[5],arra[4]],v2]=hb([arra[7],arra[6],arra[5],arra[4]])
            [[arra[11],arra[10],arra[9],arra[8]],v3]=hb([arra[11],arra[10],arra[9],arra[8]])
            [[arra[15],arra[14],arra[13],arra[12]],v4]=hb([arra[15],arra[14],arra[13],arra[12]])
            v+=v1+v2+v3+v4
            self.a=arra
            #self.rnew()
            if ifprt==1:
                print('1')
                self.prt()
            return(v)
    def prt(self):
        arra=self.a
        print('%5d,%5d,%5d,%5d'%(arra[0],arra[1],arra[2],arra[3]))
        print('%5d,%5d,%5d,%5d'%(arra[4],arra[5],arra[6],arra[7]))
        print('%5d,%5d,%5d,%5d'%(arra[8],arra[9],arra[10],arra[11]))
        print('%5d,%5d,%5d,%5d'%(arra[12],arra[13],arra[14],arra[15]))
        print('\n')
    def end(self):
        if self.avai()==[1,1,1,1]:
            return(1)
        return(0)
    def fzr(self):
        arra=self.a
        mid=np.array(arra)
        mid[mid>0]=1
        #print(mid)
        return(mid.tolist())
    def nbplay(self,i,ifprtk=0):
        if i==0:
            v=self.upus()
        if i==1:
            v=self.dpus()
        if i==2:
            v=self.lpus(ifprt=ifprtk)
        if i==3:
            v=self.rpus(ifprt=ifprtk)
        return(v)
    def alze(self):
        j=1
        arra=self.a
        for k in range(16):
            if arra[k]==0:
                j=0
        return(j)
    def played(self,prob,bug1=0):
        a=np.random.uniform()
        if a<prob[0]:
            r=self.upus()
            if bug1==1:
                print(r)
            [self,place,nb]=self.rnew(needdata=1)
            return([0,r,place,nb])
        if prob[0]<=a<prob[0]+prob[1]:
            r=self.dpus()
            if bug1==1:
                print(r)
            [self,place,nb]=self.rnew(needdata=1)
            return([1,r,place,nb])
        if prob[0]+prob[1]<=a<prob[0]+prob[1]+prob[2]:
            r=self.lpus()
            if bug1==1:
                print(r)
            [self,place,nb]=self.rnew(needdata=1)
            return([2,r,place,nb])
        r=self.rpus()
        if bug1==1:
            print(r)
        [self,place,nb]=self.rnew(needdata=1)
        return([3,r,place,nb])
    def __deepcopy__(self,hh):
        selfnew=State()
        for i in range(16):
            selfnew.a[i]=cp.deepcopy(self.a[i])
        return(selfnew)
    def __eq__(selfa,selfb):
        whe=True
        for i in range(16):
            if selfa.a[i]!=selfb.a[i]:
                whe=False
        return(whe)
    def playednb2(self,prob,bug1=0):
        a=np.random.uniform()
        b=prob.index(max(prob))
        if b==0:
            r=self.upus()
            if bug1==1:
                print(r)
            [self,place,nb]=self.rnew(needdata=1)
            return([0,r,place,nb])
        if b==1:
            r=self.dpus()
            if bug1==1:
                print(r)
            [self,place,nb]=self.rnew(needdata=1)
            return([1,r,place,nb])
        if b==2:
            r=self.lpus()
            if bug1==1:
                print(r)
            [self,place,nb]=self.rnew(needdata=1)
            return([2,r,place,nb])
        r=self.rpus()
        if bug1==1:
            print(r)
        [self,place,nb]=self.rnew(needdata=1)
        return([3,r,place,nb])
#state=State([0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0])
#state.prt()
#print(state.play([0.33,0.33,0.33,0]))

        
