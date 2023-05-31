import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft as sy
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from DataLoader import getImgs, getData, load_Vehicle_img
from torchinfo import summary as info_summary
#from torchsummaryX import summary as X_summary
from torchmetrics.functional import f1_score

class Arguments():
    def __init__(self):
        self.images = 12000 #총 이미지 수(with GAN: 12000, without: 6000)
        self.clients = 3 #총 클라이언트 수
        self.Data_Dividing_num = 3 # 전체 데이터 셋을 나누는 갯수
        self.rounds = 200 #모든 클라이언트가 한번씩 학습 한 결과를 1라운드
        self.epochs = 20 #훈련 데이터 전체를 반복하는 횟수
        self.local_batches = 20 #학습 한번에 불러올 데이터 크기
        self.lr = 0.01 #얼마나 빠르게 학습할 것인가?
        self.C = 1.0 #라운드에 얼마나 많은 클라이언트를 사용할 것인가
        self.drop_rate = 0.25
        self.torch_seed = 0 #랜덤 시드 고정
        self.log_interval = 20 #n번의 에포크 마다 학습 결과를 출력하기 위한 인자
        self.iid = 'noniidcan' #iid환경에서의 테스트를 하기 위한 인자
        self.split_size = int(self.images / self.clients) #Num of total imgs /client 수, 즉, 클라이언트 마다 할당되는 데이터의 수
        self.samples = self.split_size / self.images #논문에서 정의하고 있는 샘플 크기 (nk/n)
        self.use_cuda = True
        self.save_model = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 5, 1)  # input channels, output channels, kernel_size, stride
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 128, 5, 1)
        self.fc1 = nn.Linear(128*9*6, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(args.drop_rate)

    def forward(self, x):
        # x shape[-1, 4, 100, 80]
        x = F.relu(self.conv1(x))
        # x shape[-1, 16, 96, 76]
        x = F.max_pool2d(x, 2, 2)
        # x shape[-1, 16, 48, 38]
        x = F.relu(self.conv2(x))
        # x shape[-1, 32, 44, 34]
        x = F.max_pool2d(x, 2, 2)
        # x shape[-1, 32, 22, 17]
        x = F.relu(self.conv3(x))
        # x shape[-1, 128, 18, 13]
        x = F.max_pool2d(x, 2, 2)
        # x shape[-1, 128, 9, 6]
        x = x.view(-1, 128*9*6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x shape [-1, 512]
        x = self.fc2(x)
        # x shape [-1, 128]
        x = self.fc3(x)
        # x shape [-1, 32]
        x = self.fc4(x)
        # x shape [-1, 4]
        return F.log_softmax(x, dim=1)

def ClientUpdate(args, device, client): # 10개 중 임의로 선택된 클라이언트가 입력 파라미터로 들어온다
    client['model'].train() # client 의 model을 훈련 모드로 설정
    #client['model'].send(client['fl'])  # client의 model들, fl을 전송

    # 에포크 수만큼 반복 시키기 위해 범위를 지정했고,
    # 출력문 문제 때문에 1부터 시작하였음
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (X, y) in enumerate(client['trainset']): # X는 Real img, y는 Label
            # X와 Y를 불러오고 X와 y를 클라이언트에게 전송함
            #X = X.send(client['fl'])
            #y = y.send(client['fl'])
            X, y = X.to(device), y.to(device)
            # 학습 프로세스
            client['optim'].zero_grad()  # 그라디언트 초기화
            output = client['model'](X)  # 모델의 예측값 획득

            metric = f1_score(output, y, task="multiclass", num_classes=4)
            loss = F.cross_entropy(output, y)  # 예측과 Real label 사이의 loss 계산
            loss.backward()  # 역전파
            client['optim'].step()  # 파라미터 업데이트
            # loss를 출력하기 위한 출력문
            if batch_idx % args.log_interval == 0:
                loss = loss.data
                print('Model {} Train Epoch: {} [{}/{} ({:.0f}%)]\tF1: {:.3f}\tLoss: {:.6f}'.format(
                    client['fl'],
                    epoch, batch_idx * args.local_batches, len(client['trainset']) * args.local_batches,
                           100. * batch_idx / len(client['trainset']), metric, loss))

    #client['model'].get()

def test(args, model, device, test_loader, name, fed_round):
    model.eval()
    test_loss = 0
    correct = 0
    sum_metric = 0
    count = 0

    with torch.no_grad():
        #test 데이터 로더를 불러와서 예측해보기
        for X, y in test_loader:
            count += 1
            X, y = X.to(device), y.to(device)
            output = model(X)
            test_loss += F.cross_entropy(output, y, reduction='sum').item() #배치 로스 합
            pred = output.argmax(1, keepdim=True) # 결과 값으로 출력되는 log-probability를 클래스 숫자로 변경 [0,0,0,1,0,0,0]

            metric = f1_score(pred, y.view_as(pred), task="multiclass", num_classes=4)
            sum_metric += metric

            correct += pred.eq(y.view_as(pred)).sum().item()
    test_F1_score = sum_metric / count
    print("Test_F1_Score: ", test_F1_score)
    test_loss /= len(test_loader.dataset)
    Accuracy = 100. * correct / len(test_loader.dataset)
    print('\n',fed_round, 'round test set: Average loss for {} model: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        name, test_loss, correct, len(test_loader.dataset),
        Accuracy))
    return test_loss, Accuracy, test_F1_score


def averageModels(global_model, clients):
    client_models = [clients[i]['model'] for i in range(len(clients))]
    samples = [clients[i]['samples'] for i in range(len(clients))]
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_models[i].state_dict()[k].float() * samples[i] for i in range(len(client_models))], 0).sum(0)

    global_model.load_state_dict(global_dict)
    return global_model

##############################################################################################

print(torch.cuda.is_available())
args = Arguments()
use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("device: ", device)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#hook = sy.TorchHook(torch)

clients = []

# 3개의 사전 데이터 형태를 생성하고 value에 VritualWorkder id: 값 client+i 를 생성한다.
for i in range(args.clients):
    #clients.append({'fl': sy.VirtualWorker(hook, id="client{}".format(i + 1))})
    clients.append({'fl': sy.VirtualMachine(name="client{}".format(i + 1))})
print(clients) #3개, 각각 다른 차종으로 1개씩 지정

print("SOUL")
SOUL_train_set, SOUL_test_set, SOUL_train_group, SOUL_test_group = load_Vehicle_img(args.Data_Dividing_num, args.iid, 'SOUL')

# 클라이언트 별로 데이터로더를 생성 및 지정해 주는 단계
for idx, client in enumerate(clients):
    # SOUL Clients
    trainset_idx_list = list(SOUL_train_group[idx])
    test_idx_list = list(SOUL_test_group[idx])

    client['trainset'] = getImgs(SOUL_train_set, trainset_idx_list, args.local_batches)  # 훈련 데이터 로더
    client['testset'] = getImgs(SOUL_test_set, test_idx_list, args.local_batches)  # 테스트 데이터 로더
    client['samples'] = len(trainset_idx_list) / args.images  # 추후 사용할 samples변수 정의

print("Final Clients: ", clients)

#getImgs하기 전 데이터 모양은 데이터셋 전체이기 때문에, 데이터 로더를 사용하여 전체 테스트 데이터셋 데이터 로더를 생성
SOUL_test_loader = DataLoader(SOUL_test_set, batch_size=args.local_batches, shuffle=True)

torch.manual_seed(args.torch_seed) # innitialize w0
global_model = Net() #initialize model

info_summary(global_model, input_size=(args.local_batches, 4, 100, 80))

# 로컬 클라이언트 모델을 torch cpu에 로드 시키고 최적화함수를 해당 클라이언트 모델로 갱신하는 방식
# clients라는 사전에 모두 저장할 수 있도록 코드를 구성함
# 즉, 각각의 클라이언트마다 사용하는 DNN과 최적화 함수를 정의하는 과정!!!
for client in clients:
    torch.manual_seed(args.torch_seed)
    client['model'] = Net().to(device)
    client['optim'] = optim.SGD(client['model'].parameters(), lr=args.lr)


SOUL_Loss = []
SOUL_Acc = []
SOUL_F1 = []
round_axis = []

for fed_round in range(args.rounds):
    # number of selected clients
    m = int(max(args.C * args.clients, 1))

    # 선택된 클라이언트 집합을 생성하는 방법
    np.random.seed(fed_round)
    selected_clients_inds = np.random.choice(range(len(clients)), m, replace=False)  # 10개의 클라이언트 중 m개의 클라이언트를 선택함
    selected_clients = [clients[i] for i in selected_clients_inds]

    # 학습 진행
    # 논문에서 학습 진행에 ClientUpdate함수가 사용되기때문에 이를 구현
    for client in selected_clients: # 선택된 한개의 클라이언트에 대한 학습 모두 진행
        ClientUpdate(args, device, client)

    # 평균
    global_model = averageModels(global_model, selected_clients) # 선택된 client들 하나씩 반복해서 글로벌 모델 업데이트

    # Testing the average model
    soul_loss, soul_acc, soul_f = test(args, global_model, device, SOUL_test_loader, 'Global', fed_round)

    SOUL_Loss.append(soul_loss)
    SOUL_Acc.append(soul_acc)
    SOUL_F1.append(soul_f.to('cpu'))
    round_axis.append(fed_round)

    # Share the global model with the clients
    for client in clients:
        client['model'].load_state_dict(global_model.state_dict())

plt.plot(round_axis, SOUL_Loss, 'r-', label='SOUL LOSS')
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Fed Round")
plt.ylabel("LOSS")
plt.title("LOSS Graph")
plt.legend()
plt.savefig('LOSS_graph.png', dpi=300)
plt.close()

plt.plot(round_axis, SOUL_Acc, 'r-', label='SOUL Accuracy')
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Fed Round")
plt.ylabel("Accuracy")
plt.title("Accuracy Graph")
plt.legend()
plt.savefig('Accuracy_graph.png', dpi=300)
plt.close()

plt.plot(round_axis, SOUL_F1, 'r-', label='SOUL F1_score')
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Fed Round")
plt.ylabel("F1_score")
plt.title("F1_score Graph")
plt.legend()
plt.savefig('F1_score.png', dpi=300)
plt.close()

if (args.save_model):
    torch.save(global_model.state_dict(), "FedAvg.h5")






