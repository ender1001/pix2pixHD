import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
    
    
    def im_trans(frame,r_rotate,r_crop_size,r_image):

        im=imutils.rotate(frame,r_rotate)
        #pad with zeros
        im=im[r_crop_size:512-r_crop_size,r_crop_size:640-r_crop_size]
        im = cv2.resize(im, dsize=(640, 512), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        im = (im-r_image[0])*r_image[1]
        im=torch.tensor(im)

        return im
        
    
    def __getitem__(self, index):        
        ### input A (label maps)
        r_image=(np.random.rand(2)*0.2*2+1-aug_inten)*np.array([1041.0,147.0])
        r_crop_size=random.randint(0, 64)
        r_rotate=random.randint(0, 360)
        
        A_path = self.A_paths[index]              
        A = cv2.imread(A_path)[::2,::2]       
        A_tensor = im_trans(A,r_rotate,r_crop_size,r_image)


        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = cv2.imread(B_path)[::2,::2]   
            B_tensor = im_trans(B,r_rotate,r_crop_size,r_image)

                     

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
