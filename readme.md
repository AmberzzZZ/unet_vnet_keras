## model
    unet_original按照论文的示意图实现
    dropout: "Drop-out layers at the end of the contracting path perform further implicit data augmentation."
    pad: valid模式导致边缘损失，concatenate之前先crop特征图
    up-conv: upsampling(rise resolution) + conv(reduce channels)
    last layer: "At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes"
    这种策略主要是针对输入图片patch，带一圈上下文信息，优化分割结果
    
    unet_padding做了一点改动
    该模型针对整张图输入，使用same padding，保留边缘信息，输入输出尺寸相同，是对正幅图的预测

## todolist
    1. 目前版本是single-class、single-label的，扩展多标签、多类别版本（激活函数、custom loss
    2. padding model，输入输出尺寸一致的model
    3. 衍生model：unet++ etc.

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


