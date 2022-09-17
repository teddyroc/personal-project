# personal-project

class Train_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.img_path = df['file_name'].values
        self.target = df['label'].values
        self.transform = transform

        print(f'Train Dataset size:{len(self.img_path)}')

    def __getitem__(self, idx):
        images = sorted(glob.glob(self.img_path[idx]+'/*.tif'))
        shape = (256,256,13)    # args로 정해야할듯.
        nd = np.zeros(shape)
        for i, img_ in enumerate(images):
            img = cv2.imread(img_)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = center_rec_crop(img, 256)
            img = self.transform(img)    # 여기서 ToTensor()를 해줘서, permute
            nd[:,:,i] = img
        target = self.target[idx]
        return nd.permute((-1,0,1)), target

    def __len__(self):
        return len(self.img_path)  # file name개수가 곧 전체 length.
        
def get_loader(df, phase: str, batch_size, shuffle, num_workers, transform):
    if phase == 'test':
        dataset = Test_dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                 pin_memory=True)
        ''' batch size만큼의 batch를 반환하는데 이때 슬라이싱이 들어가니까 getitem에서 transform이 진행된 image가 반환됨.'''
        # the data loader will copy Tensors into CUDA pinned memory before returning them / CPU에 있던 dataset을 불러와서 GPU로 돌릴때 True
    elif phase == 'train':
        dataset = Train_Dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                 pin_memory=True,
                                 drop_last=False)

    else:
        dataset = Val_Dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                 pin_memory=True,
                                 drop_last=False)
    return data_loader
   
   
# 채널 별 mean 계산
def get_mean(dataset):
    ''' image have shape of (C, H, W) '''
    meanC = [np.mean(image.numpy(), axis=(1,2)) for image,_ in dataset]
    mean_v = []
    for i in range(13):
        mean_value = np.mean([m[i] for m in meanC])
        mean_v.append(mean_value)
    return mean_v

# 채널 별 str 계산
def get_std(dataset):
    stdC = [np.std(image.numpy(), axis=(1,2)) for image,_ in dataset]
    std_v = []
    for i in range(13):
        std_value = np.mean([s[i] for s in stdC])
        std_v.append(std_value)
    return std_v
    
    
    
def get_train_augmentation(dataset, ver):
    ''' if you want to use Non-Rigid Transformation Augmentation, use ver==2 '''
    if ver==1: # for validset
        transform = transforms.Compose([
#                 CustomCrop(),
                transforms.ToTensor(),    # Tensor로 되돌려줘야함.
                transforms.Normalize(mean=get_mean(dataset),
                                     std=get_std(dataset))
            ])
    return transform



# 모델에서
df_train = df_train[df_train['fold'] != args.fold].reset_index(drop=True)
df_val = df_train[df_train['fold'] == args.fold].reset_index(drop=True)
        
# Augmentation
self.train_transform = get_train_augmentation(norm_data, ver=args.aug_ver)  # 1 - 기본
self.test_transform = get_train_augmentation(norm_data, ver=1)

# TrainLoader, get_loader에 Dataset이 들어가있음.
self.train_loader = get_loader(df_train, phase='train', batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, transform=self.train_transform)
self.val_loader = get_loader(df_val, phase='val', batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, transform=self.test_transform)
