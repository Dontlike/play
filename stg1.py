import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import twodt as dt
import copy as cp
import random as rd
import matplotlib.pyplot as plt

# 超参数
BATCH_SIZE = 2000
LR = 0.000000001                   # learning rate
EPSILON = 0.9              # 最优选择动作百分比
GAMMA = 1                # 奖励递减参数
TARGET_REPLACE_ITER = 20   # Q 现实网络的更新频率
MEMORY_CAPACITY = 50     # 记忆库大小
N_STATES = 16   # 杆子能获取的环境信息数
EPI_PER_TRAIN=100#盘数
MIUA=0.2
MIUB=0.8





TRA_PEIR=10
SEARCH_TIME_EXP=50
SEARCH_LIY=150
HIGHEST_TEMPERTURE=20
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 200)
        self.fc1.weight.data.normal_(0, 0.01)   # initialization
        self.fc2 = nn.Linear(200,200)
        self.fc2.weight.data.normal_(0,0.01)   # initialization
        self.fc3 = nn.Linear(200,200)
        self.fc3.weight.data.normal_(0,0.01)
        self.fc4 = nn.Linear(200,200)
        self.fc4.weight.data.normal_(0,0.01)
        self.fc5 = nn.Linear(200,200)
        self.fc5.weight.data.normal_(0,0.01)
        self.fc6 = nn.Linear(200,200)
        self.fc6.weight.data.normal_(0,0.01)
        self.fc7 = nn.Linear(200,200)
        self.fc7.weight.data.normal_(0,0.01)
        self.fc8 = nn.Linear(200,200)
        self.fc8.weight.data.normal_(0,0.01)
        self.fc9 = nn.Linear(200,200)
        self.fc9.weight.data.normal_(0,0.01)
        self.fc10 = nn.Linear(200,20)
        self.fc10.weight.data.normal_(0,0.01) 
        self.out = nn.Linear(20,1)
        self.out.weight.data.normal_(0,0.01)   # initialization

    def forward(self, x):
        sig=nn.Sigmoid()
        bn1=nn.BatchNorm1d(100)
        bn2=nn.BatchNorm1d(20)
        x = self.fc1(x)
        #x=bn1(x)
        x = sig(x)
        y = self.fc2(x)
        #x=bn2(x)
        y=sig(y)
        y=self.fc3(y)
        y=sig(y)
        x=x+y
        y = self.fc4(x)
        #x=bn2(x)
        y=sig(y)
        y=self.fc5(y)
        y=sig(y)
        x=x+y
        y = self.fc6(x)
        #x=bn2(x)
        y=F.relu(y)
        y=self.fc7(y)
        y=F.relu(y)
        x=x+y
        y = self.fc8(x)
        #x=bn2(x)
        y=F.relu(y)
        y=self.fc9(y)
        y=F.relu(y)
        x=x+y
        x = self.fc10(x)
        #x=bn2(x)
        x=F.relu(x)
        actions_value=self.out(x)
        return actions_value
class DQN(object):
    def __init__(self):
        #self.eval_net=torch.load('evanet.pkl')
        #self.target_net =torch.load('tarnet.pkl')
        self.eval_net,self.target_net =Net(),Net()

        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES+1))     # 初始化记忆库
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.L1Loss()   # 误差公式
        self.loss=[]
        self.loss_counter=0
        self.loss_adder=0
    

    def choose_action(self, x,sraw,perc):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        srawcp=cp.deepcopy(sraw)
        # 这里只输入一个 sample
        rarr=[0,0,0,0]
        maxn=-1
        maxid=0
        avai=srawcp.avai()
        #print('avai:\n')
        #print(avai)
        #srawcp.prt()
        for i in range(4):
            if avai[i]==1:
                rarr[i]=-1
            else:
                srawcppl=cp.deepcopy(srawcp)
                r1=srawcppl.nbplay(i)
                arra=srawcppl.a
                times=0
                valsum=0
                for m in range(2):
                    j=m+1
                    for k in range(16):
                        if arra[k]==0:
                            new=srawcppl.ad(k,2*j)
                            newar=cp.deepcopy(new.a)
                            newar.shape=1,16
                            newar= Variable(torch.FloatTensor(newar))
                            times+=1
                            valsum+=r1+float(self.eval_net(newar).data.numpy()[0][0])
                            #if rd.randint(0,10000)>9990:
                                #print(self.eval_net(newar))
                                #print(self.eval_net(newar).data.numpy()[0][0])
                                #print(int(self.eval_net(newar).data.numpy()[0][0]))
                try:
                    rarr[i]=valsum/times
                except ZeroDivisionError:
                    srawcppl.prt()
                    print('i:%d\n'%(i))
                    srawcppl.nbplay(i,ifprtk=1)
                    print(srawcppl.avai())
                    srawcppl.prt()
                
                            
        if np.random.uniform() < EPSILON:   # 选最优动作
            for i in range(4):
                if rarr[i]>maxn:
                    maxid=i
                    maxn=rarr[i]
            action=maxid
            
        else:
            action = np.random.randint(0, 4)
        while avai[action]==1:
            action = np.random.randint(0, 4)
                
        return action

    def store_transition(self, s, r):
        transition=np.zeros(17)
        transition[0:16] = s[0:16]
        transition[16]=r
        #print(transition)
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter> MEMORY_CAPACITY:
            self.learn()

    def learn(self):
        #print(1)
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES:N_STATES+1]))

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s)  # shape (batch, 1)
        loss = self.loss_func(q_eval,b_r)
        self.loss_counter+=1
        self.loss_adder+=loss.data.numpy()[0]
        self.lose_counter=0
        self.lose_adder=self.loss_adder
        print(self.lose_adder)
        self.loss.append(self.lose_adder)
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(self.loss)
        

        plt.pause(0.0000001)  # pause a bit so that plots are updated
        self.loss_adder=0

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save(self):
        torch.save(self.eval_net,'evanet.pt')
        print("OK")
class Tnode:
    def __init__(self,state):
        self.state=cp.deepcopy(state)
        self.haveson=0
        self.avai=state.avai()
        self.sonedge=[]
        self.visittimes=[0.001,0.001,0.001,0.001]
        self.valueeval=[0,0,0,0]
        self.returnnow=[0,0,0,0]
        for ite in range(4):
            statenow=cp.deepcopy(state)
            if self.avai[ite]==0:
                self.returnnow[ite]=statenow.nbplay(ite)
            
    def play(self,temperture):
        v=[0,0,0,0]
        for i in range(4):
            if self.avai[i]==1:
                v[i]=0
            else:
                v[i]=self.visittimes[i]**temperture
        vsum=v[0]+v[1]+v[2]+v[3]
        return([v[0]/vsum,v[1]/vsum,v[2]/vsum,v[3]/vsum])
    def go(self,state,trend,nb,place):
        #print('state of go sraw')
        #state.prt()
        #print('state of edge')
        #self.sonedge[trend].store[nb//2-1][place].state.prt()
        if self.sonedge[trend].store[nb//2-1][place].state==state:
            self=self.sonedge[trend].store[nb//2-1][place]
            #print('go now tree a')
            #self.printtnode()
        else:
            self=Tnode(state)
            #print('go now tree b')
            #self.printtnode()
        return(self)

        
    def ifend(self):
        return(self.state.end())
    def traveltree(self,net,miu=0.2):
        if self.ifend()==1:
            return(0)
        v=0
        if self.haveson==0:
            v=self.expand(net)
            self.haveson=1
            #print(1)
            #print(v)
            #print(1)
            return(v)
        if np.random.uniform()<miu:
            maxid=0
            maxva=-1
            for iteri in range(4):
                if self.valueeval[iteri]/self.visittimes[iteri]>maxva and self.avai[iteri]!=1:
                    maxid=iteri
                    maxva=self.valueeval[iteri]/self.visittimes[iteri]
            vnow=self.sonedge[maxid].traveledge(net)
            try:
                self.valueeval[maxid]+=vnow+self.returnnow[maxid]
            except TypeError:
                print(vnow)
                print(self.returnnow[maxid])
            self.visittimes[maxid]+=1
            return(vnow+self.returnnow[maxid])
        while True:
            visitn=rd.randint(0,3)
            if self.avai[visitn]==0:
                break
        vnow=self.sonedge[visitn].traveledge(net)
        self.valueeval[visitn]+=vnow+self.returnnow[visitn]
        self.visittimes[visitn]+=1
        return(vnow+self.returnnow[visitn])
    def expand(self,net):
        v=0
        if self.avai[0]==0:
            staten=cp.deepcopy(self.state)
            staten.upus()
            self.sonedge.append(cp.deepcopy(Edge(staten)))
            v+=self.returnnow[0]+self.sonedge[0].evaluate(net)
        else:
            self.sonedge.append(0)
        if self.avai[1]==0:
            staten=cp.deepcopy(self.state)
            staten.dpus()
            self.sonedge.append(cp.deepcopy(Edge(staten)))
            v+=self.returnnow[1]+self.sonedge[1].evaluate(net)
        else:
            self.sonedge.append(1)
        if self.avai[2]==0:
            staten=cp.deepcopy(self.state)
            staten.lpus()
            self.sonedge.append(cp.deepcopy(Edge(staten)))
            v+=self.returnnow[2]+self.sonedge[2].evaluate(net)
        else:
            self.sonedge.append(2)
        if self.avai[3]==0:
            staten=cp.deepcopy(self.state)
            staten.rpus()
            self.sonedge.append(cp.deepcopy(Edge(staten)))
            v+=self.returnnow[3]+self.sonedge[3].evaluate(net)
        else:
            self.sonedge.append(3)
        return(v/4)
    def printtnode(self):
        self.state.prt()
        print(self.avai)
        i=0
        for item in self.sonedge:
            if type(item)!=type(0):
                print(i)
                item.state.prt()
            i+=1
    def __deepcopy__(self,hhh):
        statetem=dt.State()
        selfnew=Tnode(statetem)
        selfnew.state=cp.deepcopy(self.state)
        selfnew.haveson=cp.deepcopy(self.haveson)
        selfnew.avai=cp.deepcopy(self.avai)
        selfnew.visittimes=cp.deepcopy(self.visittimes)
        selfnew.valueeval=cp.deepcopy(self.valueeval)
        selfnew.returnnow=cp.deepcopy(self.returnnow)
        i=0
        for item in self.sonedge:
            selfnew.sonedge[i]=cp.deepcopy(item)
            i+=1
        return(selfnew)
        
class Edge:
    def __init__(self,state):
        self.state=cp.deepcopy(state)
        self.avai=[]
        self.store=[[],[]]
        for it in range(16):
            if state.a[it]!=0:
                self.avai.append(0)
                self.store[0].append(0)
                self.store[1].append(0)
            else:
                self.avai.append(1)
                self.store[0].append(Tnode(state.ad(it,2)))
                self.store[1].append(Tnode(state.ad(it,4)))
    def traveledge(self,net):
        adn=rd.randint(0,1)
        while True:
            adplace=rd.randint(0,15)
            if self.avai[adplace]==1:
                break
        v=self.store[adn][adplace].traveltree(net)
        return(v)
    def evaluate(self,net):
        v=0
        counter=0
        for nb in range(2):
            for place in range(16):
                if self.avai[place]==1:
                    #try:
                    statenow=cp.deepcopy(np.array(self.store[nb][place].state.a))
                    #except AttributeError:
                        #self.state.prt()
                        #print('%5d%5d'%(nb,place))
                    statenow.shape=1,16
                    b_s = Variable(torch.FloatTensor(statenow.tolist()[0:16]))
                    v_eval = net(b_s)
                    v+=v_eval.data.numpy()[0][0]
                    counter+=1
        return(v/counter)
    def __deepcopy__(self,hh):
        statetep=dt.State()
        selfnew=Edge(statetep)
        selfnew.state=cp.deepcopy(self.state)
        #print(1)
        #self.state.prt()
        #print(1)
        selfnew.avai=cp.deepcopy(self.avai)
        for i in range(2):
            j=0
            for item in self.store[i]:
                selfnew.store[i][j]=cp.deepcopy(item)
                #print(item)
                
                j+=1
        return(selfnew)
    def edgeprt(self):
        self.state.prt()
        print(self.avai)
        for i in range(2):
            for j in range(16):
                if type(self.store[i][j])==type(0):
                    print(self.store[i][j])
                else:
                    try:
                        self.store[i][j].state.prt()
                    except AttributeError:
                        print(self.store[i][j].state)
        
        
    
        
        
            
            
        
dqn = DQN() # 定义 DQN 系统
#dqn.__init__()
#dqn.eval_net=torch.load('evanet.pt')
#dqn.target_net=torch.load('tarnet.pt')

vsum=0
vsuml=[]
for i_episode in range(EPI_PER_TRAIN):
    memory=[]
    if i_episode>60:
        print('num:%5d,ave:%5f'%(i_episode,vsum/50))
        vsuml.append(vsum/50)
        vsum=0
        dqn.save()
        #print(i_episode)
        
    arra=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    sraw=dt.State(arra)
    perc=i_episode/EPI_PER_TRAIN
    sraw=sraw.rnew()
    v=0
    step_counter=0
    rootnode=Tnode(sraw)
    print('first')
    sraw.prt()
    #print('first tree')
    #rootnode.state.prt()
    #print(arra)
    #sraw=sraw.rnew()
    #sraw.prt()
    
    while True:
        s=cp.deepcopy(np.array(sraw.a))
        #sraw.prt()
        #print(s)
        s.shape=1,16
        #print('end:%4d'%(sraw.end()))
        #print(s)
        for searchtime in range(SEARCH_TIME_EXP):
            rootnode.traveltree(dqn.eval_net)
        for searchtime in range(SEARCH_LIY):
            rootnode.traveltree(dqn.eval_net,miu=MIUB)
        #sraw.prt()
        #print(sraw.a[3])
        #print(type(sraw))
        #print(rootnode.play(step_counter/HIGHEST_TEMPERTURE))
        #print(sraw.played(rootnode.play(step_counter/HIGHEST_TEMPERTURE),bug1=1))
        prob=rootnode.play(step_counter/HIGHEST_TEMPERTURE)
        print('old state')
        sraw.prt()
        print('data')
        print('visittime')
        print(rootnode.visittimes)
        print('valuesum')
        print(rootnode.valueeval)
        [trend,r,place,nb]=sraw.playednb2(prob)
        print(prob)
        print([trend,r,place,nb])
        sraw=sraw.ad(place,nb)
        print('new state')
        sraw.prt()
        rootnode=rootnode.go(sraw,trend,nb,place)
        #print('state of tree')
        3
        rootnode.state.prt()
        step_counter+=1
        
            
        
            #
        

            # 选动作, 得到环境反馈

        v+=r
        
        #print('a:%4d'%(a))
        #print('r:%4d'%(r))
        #print('v:%4d'%(v))
        #if sraw.alze()==1:
            #sraw.prt()
        for item in memory:
            item[1]+=r
        memory.append([s,r])

            # 存记忆
        

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn() # 记忆库满了就进行学习

        if sraw.end()==1:
            for item in memory:
                [snow,rnow]=item
                dqn.store_transition(snow,rnow)
            memory=[]
            vsum+=v
            print("epi:%5d,v:%5d"%(i_episode,v))# 如果回合结束, 进入下回合
            break

#plt.plot(vsuml)
#plt.ylabel('score')
#plt.show()

dqn.save()
