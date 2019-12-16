# unet
## model
    unet_original按照论文的示意图实现
    1. dropout: "Drop-out layers at the end of the contracting path perform further implicit data augmentation."
    2. pad: valid模式导致边缘损失，concatenate之前先crop特征图
    3. up-conv: upsampling(rise resolution) + conv(reduce channels)
    4. last layer: "At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes"
    5. 这种策略主要是针对输入图片patch，带一圈上下文信息，优化分割结果
    
    unet_padding做了一点改动：
    1. 该模型针对整张图输入，使用same padding，保留边缘信息，输入输出尺寸相同，是对整幅图的预测
    2. 每一个卷积层后面接一个batch normalization，替换掉原论文的dropout
    3. 卷积block可以配置成residual

## training data
    1. membrane: 细胞1细胞壁0，512*512，30train+20test，注意正负样本不均匀
    2. disc: 间盘，512*512，71train+100test

## todolist
    1. 目前版本是single-class、single-label的，扩展多标签、多类别版本（激活函数、custom loss
    2. 衍生model：unet++ etc.

## multi-class & multi-label
    multi-class的mask是multi-channel的
    那么对应地，最后1*1 conv的channel维度应该是n_classes(single-class  --->  1)
    有两种情况：
    single-label：each pixel的各类别之间应该是竞争关系，那么activation可以用softmax，计算loss时可以加入交叉熵
    multi-label：有时物体之间有重叠，部分pixel拥有multi-label，那么activation应该用sigmoid，loss考虑二元交叉熵

## reweighting:
    reweighting bce: 正负样本unbalance
    bce_dice: 两个loss调整到相当的数量级
    reweighting dice: 可以对每个通道分别计算dice，然后加权求和

## inference:
    按照原论文的模型来实现，输入输出大小不同————valid padding的时候有边缘信息损失
    在做prediction时，为了预测整幅图的分割结果，要输入比原图更大尺寸的图，多出来的部分通过镜像来补全（cv2.copyMakeBorder）。

## 论文笔记:
    https://amberzzzz.github.io/2019/12/05/unet-vnet/




# vnet
## model
    按照原论文中的结构和维度来实现，3D换成2D，不关注原图的第三维
    residual: element-wise sum
    unlinearity: PReLU，最后输出的部分有点没看懂，
    decompression: 解码的residual，一边concatenate了feature map，一边直接作为shortcut，前者channel数肯定多于后者，add之前先做zeropadding
    output Layer: 最后一层的图有点问题，看图是1*1 conv后面有个PReLU，然后再接一个softmax？？？

## 相比较于unet:
    residual
    diceloss

## focal loss nan:
    nan问题————当对过小的数值进行log操作，返回值将变为nan
    解决：clamp

## todolist:
    settings: multi-class & multi-channel & multi-label
    loss: reweighting & dice efficient
    data: preparation and augmentation
    experiment

## add & concatenate:
    add操作是by element相加，要求两个输入shape完全相同，如果不同，先zero padding
    concatenate操作是在channel维度的stack，要求两个输入其他维度的shape相同，channel维度可以不同

