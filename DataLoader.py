import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def IID(dataset, num_clients):
    num_of_images = int(len(dataset)/num_clients) #client별로 할당되는 이미지 개수 (즉, 전체 이미지 수 / client 수)
    clients_dict, indices = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        np.random.seed(i) #랜덤 시드를 고정하여
        #총 60,000개 indices에서 60000/num_clients로 나눠준 수 만큼 랜덤 균등 분포로 추출함
        clients_dict[i] = set(np.random.choice(indices, num_of_images, replace=False)) #random.choice함수가 그역할
        indices = list(set(indices) - clients_dict[i]) #선택한 인덱스에 해당하는 데이터를 추출하려는 곳에서 제거함
    return clients_dict

'''not using
def NonIID(dataset, num_clients, test=False):
    classes, images = 100, 600  # 10개의 클라이언트가 10개의 클래스를 선택하면 총 100개의 클래스들을, class당 600개 이미지씩 추출한다.
    if test:
        classes, images = 20, 500
    classes_idx = [i for i in range(classes)]  # 0~99 classes idx 생성
    clients_dict = {i: np.array([]) for i in range(num_clients)}  # 클라이언트 수만큼 생성 10개로 가정
    indices = np.arange(classes * images)  # 60,000개 이미지 순서

    unsorted_labels = dataset.train_labels.numpy()  # 훈련 데이터의 클래스 순서 그대로 가져오기

    if test:
        unsorted_labels = dataset.test_labels.numpy()

    indices_unsorted_labels = np.vstack((indices, unsorted_labels))  # 60,000개 이미지와 레이블 번호가 같이 세팅됨
    indices_labels = indices_unsorted_labels[:, indices_unsorted_labels[1, :].argsort()]  # 두번째 행을 기준으로 정렬
    indices = indices_labels[0, :]

    for i in range(num_clients):
        temp = set(np.random.choice(classes_idx, 2, replace=False))  # 2개의 추출 기준을 획득
        classes_idx = list(set(classes_idx) - temp)  # 추출된 2개 기준을 삭제
        for t in temp:
            clients_dict[i] = np.concatenate(
                (clients_dict[i], indices[t * images:(t + 1) * images]), axis=0)  # 선택된 추출 기준별 600개를 선택해서 추출
    return clients_dict
'''

def NonIID_CAN(dataset, num_clients, test=False): # 3개 클라이언트 대상으로 작성함
    classes, images = 6, 2000  # 10개의 클라이언트가 4개의 클래스를 선택, class당 00개 이미지씩 추출한다.(with gan: 6, 2000, without: 6, 1000)
    if test:
        classes, images = 25, 32 #(with gan: 25, 32, without: 25, 16)
    classes_idx = [i for i in range(classes)]  # classes idx 생성
    clients_dict = {i: np.array([]) for i in range(num_clients)}  # 클라이언트 수만큼 생성
    clients_class_dict = {i: np.array([]) for i in range(num_clients)}  # 클라이언트 각각에 어떤 클래스의 데이터가 들어갔는지 알기 위함
    indices = np.arange(classes * images)  # 2000개 이미지 순서

    unsorted_list = dataset.imgs  # 훈련 데이터 가져오기
    unsorted_labels = [] # 훈련셋 라벨을 위한 list
    for i in range(len(unsorted_list)):
        unsorted_labels.append(unsorted_list[i][1]) #훈련셋 라벨 순서대로 모두 추출
    #print("\n classes: ", dataset.class_to_idx, "\nLabels: ", unsorted_labels)

    indices_unsorted_labels = np.vstack((indices, unsorted_labels))  # 이미지와 레이블 번호가 같이 세팅됨
    indices_labels = indices_unsorted_labels[:, indices_unsorted_labels[1, :].argsort()]  # 두번째 행을 기준으로 정렬
    indices = indices_labels[0, :]

    clas = list(indices_labels[1, :])
    print("클래스별 갯수 카운팅: ", clas.count(0), clas.count(1), clas.count(2), clas.count(3))


    for i in range(num_clients):
        if test:
            N=8
        else:
            N=2
        temp = set(np.random.choice(classes_idx, N, replace=False))  # 3개의 추출 기준을 획득
        classes_idx = list(set(classes_idx) - temp)  # 추출된 N개 기준을 삭제
        for t in temp: #총 N번씩, images 갯수를 각 클라이언트마다 할당, 즉 각 클라이언트는 여기서 images * N 장씩 할당받음
            clients_dict[i] = np.concatenate(
                (clients_dict[i], indices[t * images:(t + 1) * images]), axis=0)  # 선택된 추출 기준별 images개씩을 선택해서 이미지 추출
            clients_class_dict[i] = np.concatenate(
                (clients_class_dict[i], clas[t * images:(t + 1) * images]), axis=0)  # 선택된 추출 기준별 images개씩을 선택해서 라벨 추출

    num_of_zero = 0
    num_of_one = 0
    num_of_two = 0
    num_of_three = 0
    for i in range(num_clients):
        num_of_zero += list(clients_class_dict[i]).count(0)
        num_of_one += list(clients_class_dict[i]).count(1)
        num_of_two += list(clients_class_dict[i]).count(2)
        num_of_three += list(clients_class_dict[i]).count(3)
        print("\nclient {}\nFlooding: {}\nFuzzy:  {}\nMal: {}\nNormal: {}".format(i,
                                                                                    list(clients_class_dict[i]).count(0),
                                                                                    list(clients_class_dict[i]).count(1),
                                                                                    list(clients_class_dict[i]).count(2),
                                                                                    list(clients_class_dict[i]).count(3)))
    print("\n<Total> Flooding: {}, Fuzzy: {}, Mal: {}, Normal: {}\n".format(num_of_zero, num_of_one, num_of_two, num_of_three))
    return clients_dict



def custom_loader(path): ## Loader func for RGBA channel image
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


def load_Vehicle_img(num_clients, iidtype, Vehicle_type):
    trainset = datasets.ImageFolder(root='C:/Users/YongJun/Desktop/연구/Data/Fed_Set/SOUL_one_withGAN/train/',
                                    transform=transforms.ToTensor(),
                                    loader=custom_loader
                                    )
    testset = datasets.ImageFolder(root='C:/Users/YongJun/Desktop/연구/Data/Fed_Set/SOUL_one_withGAN/test/',
                                   transform=transforms.ToTensor(),
                                   loader=custom_loader
                                    )

    train_group, test_group = None, None

    if iidtype == 'iid':
        traingroup = IID(trainset, num_clients)
        testgroup = IID(testset, num_clients)
    elif iidtype == 'noniid':
        traingroup = NonIID(trainset, num_clients)
        testgroup = NonIID(testset, num_clients, test=True)
    elif iidtype == 'noniidcan':
        traingroup = NonIID_CAN(trainset, num_clients)
        testgroup = NonIID_CAN(testset, num_clients, test=True)

    return trainset, testset, traingroup, testgroup


class FedDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        images, label = self.dataset[self.idx[item]]
        return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()


def getImgs(dataset, indices, batch_size):
    return DataLoader(FedDataset(dataset, indices), batch_size=batch_size, shuffle=True)

def getData(dataset, indices):
    return FedDataset(dataset, indices)