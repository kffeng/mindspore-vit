import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp
from mindspore import Tensor, context

# context.set_context(mode=context.GRAPH_MODE)

def drop_path(x, drop_prob: float=0., training: bool=False):

    if drop_prob==0. or training:
        return x
    keep_prob=1-drop_prob
    shape=(x.shape[0],)+(1,)*(x.ndim-1)
    random_tensor=keep_prob+mnp.rand(shape,dtype=x.dtype)
    random_tensor=P.floor(random_tensor)
    output=P.div(x,keep_prob)*random_tensor
    return output



class DropPath(nn.Cell):
    def __init__(self,drop_prob=None):
        super(DropPath,self).__init__()
        self.drop_prob=drop_prob

    def construct(self,x):
        return drop_path(x,self.drop_prob,self.training)


class PatchEmbed(nn.Cell):
    def __init__(self,img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None,dtype=None):
        # 我们调用了 super().__init__() 来确保父类的初始化方法被执行
        # 在不完全覆盖父类方法的情况下，允许您扩展父类中的方法。
        super().__init__()

        img_size=(img_size,img_size)
        patch_size=(patch_size,patch_size)
        self.img_size=img_size
        self.patch_size=patch_size
        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches=self.grid_size[0]*self.grid_size[1]

        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size).to_float(dtype)
        self.norm=norm_layer((embed_dim,)) if norm_layer else nn.Identity()

        self.embed_dim=embed_dim


    def construct(self,x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x=self.proj(x)  # B, 768,14,14
        x=x.reshape(B,self.embed_dim,self.num_patches).transpose(0,2,1)
        x=self.norm(x)
        return x


class Attention(nn.Cell):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bais=False,
                 qk_scale=None,
                 attn_drop_ration=0.,
                 proj_drop_ration=0.,
                 dtype=None):
        super(Attention,self).__init__()
        self.num_heads=num_heads
        head_dim=dim//num_heads
        self.scale=qk_scale or head_dim**-0.5
        self.qkv=nn.Dense(dim,dim*3,has_bias=qkv_bais).to_float(dtype)
        self.attn_drop=nn.Dropout(keep_prob=1-attn_drop_ration)
        self.proj=nn.Dense(dim,dim).to_float(dtype)
        self.proj_drop=nn.Dropout(keep_prob=1-proj_drop_ration)
        self.BatchMatMul1=P.BatchMatMul(transpose_b=True)
        self.softmax=nn.Softmax(axis=-1).to_float(ms.float32)
        self.BatchMatMul2=P.BatchMatMul()
        self.cast=P.Cast()
        self.dtype=dtype



    def construct(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).transpose(2,0,3,1,4)
        q, k, v=qkv[0], qkv[1], qkv[2]

        # B,num_heads,N,N

        attn=self.BatchMatMul1(q,k)*self.scale
        attn=self.softmax(attn)
        attn=self.attn_drop(attn)

        attn=self.cast(attn,self.dtype)

        #  B,num_heads,N,C//num_heads--->B,N,num_heads,C//num_heads
        x=self.BatchMatMul2(attn,v).transpose(0,2,1,3).reshape(B,N,C)
        x=self.proj(x)
        x=self.proj_drop(x)
        return x





class Mlp(nn.Cell):
    def __init__(self,in_features,hidden_features=None,out_features=None,dtype=None,act_layer=nn.GELU,drop=0.,):
        super().__init__()
        out_features=out_features or in_features
        hidden_features=hidden_features or in_features
        self.fc1=nn.Dense(in_features,hidden_features).to_float(dtype)
        self.act=act_layer()
        self.fc2=nn.Dense(hidden_features,out_features).to_float(dtype)
        self.drop=nn.Dropout(keep_prob=1-drop)

    def construct(self,x):
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x



class Block(nn.Cell):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ration=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ration=0.,
                 attn_drop_ration=0.,
                 drop_path_ration=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 dtype=None):
        super(Block,self).__init__()

        self.norm1=norm_layer((dim,))
        self.attn=Attention(dim=dim,num_heads=num_heads,qkv_bais=qkv_bias,qk_scale=qk_scale,
                            attn_drop_ration=attn_drop_ration,proj_drop_ration=drop_ration,dtype=dtype)
        self.drop_path=DropPath(drop_path_ration) if drop_path_ration > 0. else nn.Identity()
        self.norm2=norm_layer((dim,))
        self.mlp=Mlp(in_features=dim,hidden_features=int(mlp_ration*dim),dtype=dtype,act_layer=act_layer,drop=drop_ration,)


    def construct(self, x):
        '''
        结合drop_path的调用,若x为输入的张量,其通道为[B,C,H,W],那么drop_path的含义为在一个Batch_size中,
        随机有drop_prob的样本,不经过主干,而直接由分支进行恒等映射。
        '''
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x



class VisionTransformer(nn.Cell):
    def __init__(self,img_size=224,patch_size=16,in_c=3,num_classes=1000,
                 embed_dim=768,depth=12,num_heads=12,mlp_ration=4.0,qkv_bias=True,
                 qk_scale=None,represention_size=None,distilled=False,drop_ration=0.,
                 attn_drop_ration=0.,drop_path_ration=0.1,embed_layer=PatchEmbed,norm_layer=nn.LayerNorm,
                 act_layer=None,dtype=None):
        
        '''
        drop_ration
        attn_drop_ration
        drop_path_ration
        '''
        super(VisionTransformer,self).__init__()
        self.num_classes=num_classes
        act_layer=act_layer or nn.GELU
        dtype=dtype     

        self.num_tokens=2 if distilled else 1

        self.patch_embed=embed_layer(img_size=img_size,patch_size=patch_size,in_c=in_c,embed_dim=embed_dim,dtype=dtype)
        num_patches=self.patch_embed.num_patches

        dtype2=ms.float32
        # 在改为半精度的时候，这些初始化的参数不能为float32
        self.cls_token=ms.Parameter(P.Zeros()((1,1,embed_dim),dtype2))
        self.dist_token=ms.Parameter(P.Zeros()((1,1,embed_dim),dtype2)) if distilled else None
        self.pos_embed=ms.Parameter(P.Zeros()((1,num_patches+self.num_tokens,embed_dim),dtype2))

        self.pos_drop=nn.Dropout(keep_prob=1-drop_ration)

        dpr=[x.item(0) for x in P.linspace(Tensor(0,dtype=ms.float32),Tensor(drop_path_ration,ms.float32),depth)]
        self.blocks=nn.SequentialCell([
            Block(dim=embed_dim,num_heads=num_heads,mlp_ration=mlp_ration,qkv_bias=qkv_bias,qk_scale=qk_scale,
                  drop_ration=drop_ration,attn_drop_ration=attn_drop_ration,drop_path_ration=dpr[i],
                  act_layer=act_layer,norm_layer=norm_layer,dtype=dtype)
            for i in range(depth)
        ])

        self.norm=norm_layer((embed_dim,))

        self.head=nn.Dense(embed_dim,num_classes).to_float(dtype) if num_classes>0 else nn.Identity()

        self.cast=P.Cast()
        self.dtype=dtype
        



    def forward_feature(self,x):

        # x: B,C,H,W -->  B, 196,768
        x=self.patch_embed(x)
        cls_token=P.broadcast_to(self.cls_token,(x.shape[0],-1,-1)).astype(self.dtype)
        x=P.concat((cls_token,x),axis=1)   # B, 197, 768

        x=self.pos_drop(x+self.pos_embed.astype(self.dtype))


        x=self.blocks(x)
        x=self.norm(x)

        return x[:,0]



    def construct(self,x):
        x=self.forward_feature(x)
        # x=self.cast(x,ms.float32)
        x=self.head(x)
        return x


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool=True, ):


    model=VisionTransformer(img_size=224,
                            patch_size=16,
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            represention_size=768 if has_logits else None,
                            num_classes=num_classes,
                            dtype=ms.float16,  # 选择是单精度还是半精度，测试过程中发现两者差别很大
                            )
    return model


if __name__=="__main__":

    model=vit_base_patch16_224_in21k()
    x=mnp.rand((4,3,224,224))
    out=model(x)
    print(out)