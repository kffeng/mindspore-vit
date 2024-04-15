import argparse
import os
from utils.util import cloud_context_init,read_split_data
from utils.logger import get_logger
from utils.lr_generator import LearningRate
from utils.monitor import LossMonitor,StopAtStep
from utils.config_parser import get_config
from datasets.img_dataset import create_dataset 
from models.vit_model import vit_base_patch16_224_in21k
import mindspore.ops as P
import mindspore.nn as nn
import mindspore as ms

class NetWithLoss(nn.Cell):
    def __init__(self,net, loss):
        super().__init__()
        self.net=net
        self.loss=loss
    def construct(self,imgs, labels):
        logits=self.net(imgs)
        loss=self.loss(logits,labels)
        return loss

class EvalNet(nn.Cell):
    def __init__(self,net,loss):
        super().__init__()
        self.net=net
        self.loss=loss
    def construct(self,imgs, labels):
        logits=self.net(imgs)
        loss=self.loss(logits,labels)
        return loss,logits,labels

def main(args):

    context_config = {
        "mode": args.mode,
        "device_target": args.device_target,
        "device_id": args.device_id,
        'max_call_depth': args.max_call_depth,
        'save_graphs': args.save_graphs,
    }
    parallel_config = {
        'parallel_mode': args.parallel_mode,
        'gradients_mean': args.gradients_mean,
    }


    local_rank, device_id, device_nums=cloud_context_init(seed=args.seed,
                                                         use_parallel=args.use_parallel,
                                                         context_config=context_config,
                                                         parallel_config=parallel_config)
    
    args.device_nums = device_nums
    args.local_rank = local_rank
    args.logger = get_logger(args.save_dir)
    args.logger.info(f"local_rank: {local_rank}, device_num: {device_nums}, device_id: {device_id}")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # profiler=ms.Profiler(outout_path='./profiler_data')


    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    train_dataset=create_dataset(args=args,data_path=train_images_path,images_label=train_images_label,mode='train',shuffle=True)
    train_size=train_dataset.get_dataset_size()
    args.logger.info("Create train dataset finish, data size: {}".format(train_size))
    val_dataset=create_dataset(args=args,data_path=val_images_path,images_label=val_images_label,mode='val',shuffle=True)
    val_size=val_dataset.get_dataset_size()
    args.logger.info("Create val dataset finish, data size: {}".format(val_size))

    model=vit_base_patch16_224_in21k(num_classes=args.num_classes,has_logits=False)
    args.logger.info("Create model finish...")

    size=P.Size()
    n_parameters=sum(size(p) for p in model.trainable_params() if p.requires_grad)
    args.logger.info("number of params: {}".format(n_parameters))

    lr_schedule=LearningRate(args.start_lr, args.end_lr, args.epochs, args.warmup_epochs,train_size)

    train_loss=nn.CrossEntropyLoss()
    val_loss=nn.CrossEntropyLoss()

    net_with_loss=NetWithLoss(model,train_loss)

    optimizer=nn.SGD(params=net_with_loss.trainable_params(),
                    learning_rate=lr_schedule,
                    momentum=0.9,
                    weight_decay=5.e-5
                    )
    
    # optimizer=nn.AdamWeightDecay(params=net_with_loss.trainable_params,
    #                 learning_rate=lr_schedule,
    #                 )
    
    train_model=nn.TrainOneStepCell(network=net_with_loss,optimizer=optimizer)
    val_model=EvalNet(model,val_loss)

    # 这个可以收集损失等
    # summary_collector=ms.SummaryCollector(summary_dir='./summary_dir',collect_freq=1,)
    # 性能优化收集算子运行时长
    # profile_sum=StopAtStep(2,5)
    callbacks=[LossMonitor(per_print_times=train_size,ifeval=True,log=args.logger),]

    save_ckpt_feq=args.save_ckpt_epochs*train_size
    if local_rank==0:
        cp_config=ms.CheckpointConfig(save_checkpoint_steps=save_ckpt_feq,keep_checkpoint_max=10,)
        save_cp=ms.ModelCheckpoint(prefix='vit_base',directory=args.save_dir,config=cp_config)
        callbacks+=[save_cp,]

    # 不太理解
    model=ms.Model(network=train_model,loss_fn=None,optimizer=None,
                   metrics={"acc1": nn.TopKCategoricalAccuracy(1), "acc5": nn.TopKCategoricalAccuracy(5)},eval_network=val_model,
                   eval_indexes=[0,1,2])


    #边训练边推理
    model.fit(epoch=args.epochs,train_dataset=train_dataset,valid_dataset=val_dataset,
              callbacks=callbacks,dataset_sink_mode=args.sink_mode,sink_size=train_size)

    args.logger.info("Train Finished !!!")



if __name__=='__main__':
    parser=argparse.ArgumentParser("vision_transformer",add_help=False)
    parser.add_argument('--config_file',type=str,default="config/vision_transformer_B.yaml")
    
    args=parser.parse_args()
    args=get_config(args.config_file)

    main(args)