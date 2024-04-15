from PIL import Image
import multiprocessing
from mindspore.dataset import transforms, vision, DistributedSampler,GeneratorDataset
import mindspore as ms

class MyDataSet:
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)


        return img, label


def create_dataset(args,data_path,images_label,mode,shuffle=True):

    cores=multiprocessing.cpu_count()
    args.logger.info("cores:{}".format(cores))


    
    column_names=['img','label']
    if mode=='train':
        image_dataset=MyDataSet(images_path=data_path,
                            images_class=images_label)
        data_transform=transforms.Compose([ vision.RandomResizedCrop(224),
                                            vision.RandomHorizontalFlip(),
                                            vision.Normalize([0.5,0.5 ,0.5],[0.5,0.5 ,0.5]),
                                            vision.ToTensor(),
                                            ])
    if mode=='val':
        image_dataset=MyDataSet(images_path=data_path,
                            images_class=images_label)
        data_transform=transforms.Compose([ vision.Resize(256),
                                            vision.CenterCrop(224),
                                            vision.Normalize([0.5,0.5 ,0.5],[0.5,0.5 ,0.5]),
                                            vision.ToTensor(),
                                            ])
        
    type_cast_op=transforms.TypeCast(ms.int32)
    sampler=DistributedSampler(args.device_nums, args.local_rank, shuffle=shuffle)


    data=GeneratorDataset(source=image_dataset,
                          column_names=column_names,
                          num_parallel_workers=8,
                          python_multiprocessing=False,
                          sampler=sampler)
    
    if mode=="train":
        data=data.map(operations=data_transform,
                      num_parallel_workers=8)
        data=data.map(operations=type_cast_op,
                      input_columns=['label'],
                      num_parallel_workers=2)
    if mode=="val":
        data=data.map(operations=data_transform,
                      num_parallel_workers=4)
        data=data.map(operations=type_cast_op,
                      input_columns=['label'],
                      num_parallel_workers=2)
        
    data=data.batch(batch_size=args.batch_size,
                    drop_remainder=True)
    

    return data